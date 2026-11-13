use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use derive_more::Debug;
use thiserror::Error;

mod coarse;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, Context,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::matmul::{MatmulB, QuantParams, QuantParamsLayout, QuantizedB, QuantizedCorrection},
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum WeightMatrixError<B: Backend> {
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Unsupported weight matrix configuration: {0}")]
    UnsupportedConfiguration(String),
}

#[derive(Clone, Copy)]
pub struct QuantizationInfo {
    pub mode: QuantizationMode,
    pub method: QuantizationMethod,
    pub group_size: u32,
}

pub struct ParsedWeightSpec {
    pub layout: Layout,
    pub quantization: Option<QuantizationInfo>,
}

pub fn parse_spec<B: Backend>(spec: &AnyWeightMatrixSpec) -> Result<ParsedWeightSpec, WeightMatrixError<B>> {
    let (layout, quantized) = match spec {
        AnyWeightMatrixSpec::FullPrecisionSpec(spec) => (spec.layout.clone(), None),
        AnyWeightMatrixSpec::MLXSpec(spec) => {
            (spec.layout.clone(), Some((spec.bits, spec.group_size, QuantizationMethod::ScaleBias)))
        },
        AnyWeightMatrixSpec::IntSpec(spec) => (
            spec.layout.clone(),
            Some((
                spec.bits,
                spec.group_size,
                if spec.is_symmetric {
                    QuantizationMethod::ScaleSymmetric
                } else {
                    QuantizationMethod::ScaleZeroPoint
                },
            )),
        ),
        spec => return Err(WeightMatrixError::UnsupportedConfiguration(format!("{spec:?}"))),
    };
    let quantization = match quantized {
        None => None,
        Some((bits, group_size, method)) => {
            let mode = match bits {
                4 => QuantizationMode::U4,
                5 if method == QuantizationMethod::ScaleBias => QuantizationMode::U5,
                6 if method == QuantizationMethod::ScaleBias => QuantizationMode::U6,
                8 => QuantizationMode::U8,
                _ => {
                    return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                        "{method} bits={bits}, group_size={group_size}"
                    )));
                },
            };
            if group_size == 0 {
                return Err(WeightMatrixError::UnsupportedConfiguration("group size must be non-zero".into()));
            }
            if matches!(mode, QuantizationMode::U5 | QuantizationMode::U6)
                && (group_size != 64 || layout != Layout::OutputInput)
            {
                return Err(WeightMatrixError::UnsupportedConfiguration(
                    "packed five/six-bit affine matrices require group64 and output-input layout".into(),
                ));
            }
            Some(QuantizationInfo {
                mode,
                method,
                group_size,
            })
        },
    };
    Ok(ParsedWeightSpec {
        layout,
        quantization,
    })
}

struct Quantized<B: Backend> {
    scales: Arc<B::GlobalBuffer>,
    correction: QuantizedCorrection<Arc<B::GlobalBuffer>>,
    params: QuantParams,
    info: QuantizationInfo,
    signed_codes: bool,
}

pub struct WeightMatrix<B: Backend> {
    values: Arc<B::GlobalBuffer>,
    quantized: Option<Quantized<B>>,
    representation_copy_bytes: Arc<AtomicUsize>,
}

impl<B: Backend> WeightMatrix<B> {
    pub fn load(
        tree: &ParameterTree<B>,
        spec: AnyWeightMatrixSpec,
        required_layout: Layout,
        output_dim: u32,
        input_dim: u32,
        data_type: DataType,
    ) -> Result<Self, WeightMatrixError<B>> {
        let materialize_affine_input_sum = matches!(&spec,
            AnyWeightMatrixSpec::MLXSpec(spec) if spec.materialize_affine_input_sum);
        if materialize_affine_input_sum
            && !matches!(&spec, AnyWeightMatrixSpec::MLXSpec(spec) if spec.bits == 4 && data_type == DataType::BF16)
        {
            return Err(WeightMatrixError::UnsupportedConfiguration(
                "materialized affine input sums require BF16 W4 matrices".into(),
            ));
        }
        let ParsedWeightSpec {
            layout,
            quantization: quantization_info,
        } = parse_spec(&spec)?;
        if layout != required_layout {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "expected {required_layout:?} weight layout, got {layout:?}"
            )));
        }
        let (rows, columns) = physical_shape(&layout, output_dim, input_dim);

        let Some(info) = quantization_info else {
            let values = tree.leaf("weights")?.validate(&[rows, columns], data_type)?.read_shared_weight()?;
            return Ok(Self {
                values,
                quantized: None,
                representation_copy_bytes: tree.representation_copy_counter(),
            });
        };
        // Parameters swap the weight axes once K is grouped: output-input stores [G, N], input-output stores [N, G].
        let params_layout = match layout {
            Layout::OutputInput => QuantParamsLayout::GroupOutput,
            Layout::InputOutput => QuantParamsLayout::OutputGroup,
        };

        let group_size = info.group_size;
        if matches!(info.mode, QuantizationMode::U5 | QuantizationMode::U6)
            && (layout != Layout::OutputInput
                || group_size != 64
                || !columns.is_multiple_of(group_size)
                || data_type != DataType::BF16)
        {
            return Err(WeightMatrixError::UnsupportedConfiguration(
                "packed five/six-bit affine matrices require output-input layout, complete group-64 rows and BF16 coefficients".into(),
            ));
        }
        let storage_data_type = info.mode.storage_type();
        if !columns.is_multiple_of(if info.mode == QuantizationMode::U4 {
            2
        } else {
            1
        }) {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "stored columns {columns} do not contain complete packed bytes"
            )));
        }
        let groups = columns.div_ceil(group_size);

        let row_bytes = info
            .mode
            .packed_row_bytes(columns)
            .ok_or_else(|| WeightMatrixError::UnsupportedConfiguration("packed row size overflow".into()))?;
        let values = tree.leaf("weights")?.validate(&[rows, row_bytes], storage_data_type)?.read_shared_weight()?;
        let params = QuantParams::new(params_layout, rows, groups)
            .with_materialized_affine_input_sum(materialize_affine_input_sum);
        let load_plane = |name: &str,
                          shape: [u32; 2],
                          storage_type: DataType|
         -> Result<Arc<B::GlobalBuffer>, WeightMatrixError<B>> {
            Ok(tree.leaf(name)?.validate(&shape, storage_type)?.read_shared_weight()?)
        };
        let scales = load_plane("scales", params.scale_shape(), data_type)?;
        let correction = match info.method {
            QuantizationMethod::ScaleBias => {
                QuantizedCorrection::Biases(load_plane("biases", params.scale_shape(), data_type)?)
            },
            QuantizationMethod::ScaleZeroPoint => QuantizedCorrection::ZeroPoints(load_plane(
                "zero_points",
                params.zero_point_shape(info.mode),
                info.mode.storage_type(),
            )?),
            QuantizationMethod::ScaleSymmetric => QuantizedCorrection::Symmetric,
        };

        Ok(Self {
            values,
            representation_copy_bytes: tree.representation_copy_counter(),
            quantized: Some(Quantized {
                scales,
                correction,
                params,
                info,
                signed_codes: false,
            }),
        })
    }

    pub fn values(&self) -> &B::GlobalBuffer {
        self.values.as_ref()
    }

    pub fn quantization(&self) -> Option<QuantizationInfo> {
        self.quantized.as_ref().map(|quantized| quantized.info)
    }

    pub fn scales(&self) -> Option<&B::GlobalBuffer> {
        self.quantized.as_ref().map(|quantized| quantized.scales.as_ref())
    }

    pub fn zero_points(&self) -> Option<&B::GlobalBuffer> {
        self.quantized.as_ref()?.correction.zero_points().map(Arc::as_ref)
    }

    pub fn biases(&self) -> Option<&B::GlobalBuffer> {
        self.quantized.as_ref()?.correction.biases().map(Arc::as_ref)
    }

    pub fn matmul_b(&self) -> MatmulB<&B::GlobalBuffer> {
        let Some(quantized) = self.quantized.as_ref() else {
            return MatmulB::FullPrecision {
                b: self.values.as_ref(),
            };
        };
        let mode = quantized.info.mode;
        let group_size = quantized.info.group_size;
        let signed_codes = quantized.signed_codes;
        MatmulB::Quantized(QuantizedB {
            codes: self.values.as_ref(),
            scales: quantized.scales.as_ref(),
            correction: match &quantized.correction {
                QuantizedCorrection::Symmetric => QuantizedCorrection::Symmetric,
                QuantizedCorrection::Biases(bytes) => QuantizedCorrection::Biases(bytes.as_ref()),
                QuantizedCorrection::ZeroPoints(bytes) => QuantizedCorrection::ZeroPoints(bytes.as_ref()),
            },
            params: quantized.params,
            mode,
            group_size,
            signed_codes,
        })
    }

    pub fn try_prepare_a8_storage(
        &mut self,
        context: &B::Context,
    ) -> Result<bool, WeightMatrixError<B>> {
        let Some(quantized) = self.quantized.as_mut() else {
            return Ok(false);
        };
        if matches!(quantized.info.mode, QuantizationMode::U3 | QuantizationMode::U5 | QuantizationMode::U6)
            || quantized.params.materializes_affine_input_sum()
            || quantized.params.layout() != QuantParamsLayout::GroupOutput
        {
            return Ok(false);
        }
        if quantized.info.mode != QuantizationMode::U4 && !quantized.signed_codes {
            if let Some(sign_flip_mask) = quantized.info.mode.weight_codes_sign_flip_mask() {
                if Arc::get_mut(&mut self.values).is_none() {
                    // The image reference graph still reads unsigned source codes.
                    // Never rewrite its shared bytes when the text graph prepares A8.
                    let mut copy = context.create_buffer(self.values.as_ref().size()).map_err(|error| {
                        WeightMatrixError::ParameterError(ParameterLoaderError::BackendError(error))
                    })?;
                    copy.as_slice_mut::<u8>().copy_from_slice(self.values.as_ref().as_slice::<u8>());
                    self.representation_copy_bytes.fetch_add(self.values.as_ref().size(), Ordering::Relaxed);
                    self.values = Arc::new(copy);
                }
                let values = Arc::get_mut(&mut self.values).unwrap();
                let broadcast_mask = u64::from(sign_flip_mask) * 0x0101_0101_0101_0101;
                let (prefix, words, suffix) = bytemuck::pod_align_to_mut::<u8, u64>(values.as_slice_mut());
                words.iter_mut().for_each(|word| *word ^= broadcast_mask);
                prefix.iter_mut().chain(suffix.iter_mut()).for_each(|code| *code ^= sign_flip_mask);
            }
            quantized.signed_codes = true;
        }
        Ok(true)
    }

    pub fn set_source_output_widths(
        &mut self,
        widths: &[u32],
    ) -> Result<(), WeightMatrixError<B>> {
        if let Some(quantized) = self.quantized.as_mut().filter(|matrix| matrix.params.materializes_affine_input_sum())
        {
            quantized.params = quantized
                .params
                .with_source_output_widths(widths)
                .map_err(|message| WeightMatrixError::UnsupportedConfiguration(message.into()))?;
        }
        Ok(())
    }

    pub fn a8_signed_codes(&self) -> Option<bool> {
        self.quantized.as_ref().and_then(|quantized| match quantized.info.mode {
            QuantizationMode::U4 => Some(false),
            QuantizationMode::I8 | QuantizationMode::U8 => Some(true),
            QuantizationMode::U3 | QuantizationMode::U5 | QuantizationMode::U6 => None,
        })
    }
}

fn physical_shape(
    layout: &Layout,
    output_dim: u32,
    input_dim: u32,
) -> (u32, u32) {
    match layout {
        Layout::OutputInput => (output_dim, input_dim),
        Layout::InputOutput => (input_dim, output_dim),
    }
}

#[cfg(all(test, feature = "cpu"))]
mod packed_affine_tests {
    use stark_engine_macros::stark_test;

    use super::*;
    use crate::backends::cpu::Cpu;

    #[stark_test]
    fn packed_affine_spec_admission_is_narrow() {
        for bits in [5, 6] {
            let valid =
                serde_json::json!({"type": "MLXSpec", "bits": bits, "group_size": 64, "layout": "output_input"});
            let spec = serde_json::from_value::<AnyWeightMatrixSpec>(valid.clone()).unwrap();
            let parsed = parse_spec::<Cpu>(&spec).unwrap();
            let info = parsed.quantization.unwrap();
            assert_eq!(info.mode.bits(), bits);
            assert_eq!(info.mode.packed_row_bytes(5120), Some(5120 * bits / 8));
            assert_eq!(info.mode.storage_type(), DataType::U8);
            assert_eq!(info.mode.packed_row_bytes(u32::MAX), None);
            for (field, value) in [("group_size", serde_json::json!(32)), ("layout", serde_json::json!("input_output"))]
            {
                let mut invalid = valid.clone();
                invalid[field] = value;
                let spec = serde_json::from_value::<AnyWeightMatrixSpec>(invalid).unwrap();
                assert!(parse_spec::<Cpu>(&spec).is_err());
            }
            let mut invalid = valid;
            invalid["type"] = serde_json::json!("IntSpec");
            invalid["is_symmetric"] = serde_json::json!(true);
            let spec = serde_json::from_value::<AnyWeightMatrixSpec>(invalid).unwrap();
            assert!(parse_spec::<Cpu>(&spec).is_err());
        }
    }

    #[stark_test]
    fn internal_three_bit_storage_is_not_a_source_spec() {
        for ty in ["MLXSpec", "IntSpec"] {
            let mut encoded = serde_json::json!({
                "type": ty, "bits": 3, "group_size": 64, "layout": "output_input"
            });
            if ty == "IntSpec" {
                encoded["is_symmetric"] = serde_json::json!(true);
            }
            let spec = serde_json::from_value::<AnyWeightMatrixSpec>(encoded).unwrap();
            assert!(parse_spec::<Cpu>(&spec).is_err());
        }
    }
}
