//! Independent scalar control for production Q4/group64 expert-bank gathers.

use std::{
    fs::{File, OpenOptions},
    io::Write,
    sync::atomic::{AtomicU64, Ordering},
};

use half::bf16;
use stark_engine_macros::stark_test;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        Kernels,
        kernel::matmul::{MatmulA, MatmulArguments, MatmulDOps, MatmulKernel},
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    encodable_block::weight_matrix::WeightMatrix,
    parameters::ParameterLoader,
};

const EXPERTS: usize = 5;
const OUTPUTS: usize = 13;
const REDUCTION: usize = 256;
const GROUP_SIZE: usize = 64;
const GROUPS: usize = REDUCTION / GROUP_SIZE;
const BANK_ROWS: usize = EXPERTS * OUTPUTS;
const STORAGE_ROWS: usize = 68;
const TOKENS: usize = 5;
const TOP_K: usize = 3;
const INPUT_ROWS: usize = TOKENS * TOP_K;

struct Fixture {
    file: File,
    codes: Vec<u8>,
    // Semantic output-major parameters are separate from the on-disk
    // GroupOutput plane and never use the engine's stride/index helpers.
    affine: Vec<[bf16; 2]>,
    input: Vec<bf16>,
    experts: Vec<usize>,
    gather: Vec<u32>,
}

impl Fixture {
    fn new() -> Self {
        let mut codes = vec![0_u8; BANK_ROWS * REDUCTION / 2];
        let mut affine = Vec::new();
        for expert in 0..EXPERTS {
            for output in 0..OUTPUTS {
                let row = expert * OUTPUTS + output;
                for column in 0..REDUCTION {
                    let q = ((row * 7 + column * 3 + column / 11 + (column / GROUP_SIZE) * 5) % 16) as u8;
                    codes[row * (REDUCTION / 2) + column / 2] |= q << ((column % 2) * 4);
                }
                for group in 0..GROUPS {
                    let power = (1_u32 << group) as f32;
                    let scale = bf16::from_f32((128 + row) as f32 * power / 65_536.0);
                    let sign = if group % 2 == 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let bias = bf16::from_f32((193 - row) as f32 * power * sign / 131_072.0);
                    affine.push([scale, bias]);
                }
            }
        }
        let mut scales = vec![bf16::ZERO; GROUPS * STORAGE_ROWS];
        let mut biases = vec![bf16::ZERO; GROUPS * STORAGE_ROWS];
        for group in 0..GROUPS {
            for row in 0..BANK_ROWS {
                scales[group * STORAGE_ROWS + row] = affine[row * GROUPS + group][0];
                biases[group * STORAGE_ROWS + row] = affine[row * GROUPS + group][1];
            }
        }
        let input = (0..INPUT_ROWS * REDUCTION)
            .map(|index| {
                let row = index / REDUCTION;
                let column = index % REDUCTION;
                let numerator = ((row * 13 + column * 5 + column / 7) % 17) as i32 - 8;
                bf16::from_f32(numerator as f32 / 64.0)
            })
            .collect();
        let experts: Vec<usize> =
            (0..INPUT_ROWS).map(|row| ((row / TOP_K) * 2 + (row % TOP_K) * 3 + 1) % EXPERTS).collect();
        let gather = experts
            .iter()
            .flat_map(|expert| (0..OUTPUTS).map(move |output| (expert * OUTPUTS + output) as u32))
            .collect();
        let code_end = codes.len();
        let scale_end = code_end + scales.len() * 2;
        let bias_end = scale_end + biases.len() * 2;
        let header = serde_json::json!({
            "bank.weights":{"dtype":"U8","shape":[BANK_ROWS,REDUCTION/2],"data_offsets":[0,code_end]},
            "bank.scales":{"dtype":"BF16","shape":[GROUPS,STORAGE_ROWS],"data_offsets":[code_end,scale_end]},
            "bank.biases":{"dtype":"BF16","shape":[GROUPS,STORAGE_ROWS],"data_offsets":[scale_end,bias_end]},
        });
        let mut header = serde_json::to_vec(&header).unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "stark-expert-gather-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&codes).unwrap();
        for plane in [&scales, &biases] {
            for value in plane {
                file.write_all(&value.to_bits().to_le_bytes()).unwrap();
            }
        }
        Self {
            file,
            codes,
            affine,
            input,
            experts,
            gather,
        }
    }

    fn scalar(
        &self,
        input_row: usize,
        output: usize,
    ) -> (f64, f64) {
        let expert = self.experts[input_row];
        let mut total = 0.0_f64;
        let mut absolute_terms = 0.0_f64;
        for column in 0..REDUCTION {
            // Hand-unpack low nibble first from the canonical row code bytes.
            // Parameters use semantic coordinates, never GroupOutput offsets.
            let byte = self.codes[(expert * OUTPUTS + output) * (REDUCTION / 2) + column / 2];
            let code = if column % 2 == 0 {
                byte & 15
            } else {
                byte >> 4
            };
            let [scale, bias] = self.affine[(expert * OUTPUTS + output) * GROUPS + column / GROUP_SIZE];
            let weight = f64::from(code) * f64::from(scale.to_f32()) + f64::from(bias.to_f32());
            let term = f64::from(self.input[input_row * REDUCTION + column].to_f32()) * weight;
            total += term;
            absolute_terms += term.abs();
        }
        (total, absolute_terms)
    }
}

fn check<B: Backend>() {
    let fixture = Fixture::new();
    let context = B::Context::new().unwrap_or_else(|_| panic!("expert gather context unavailable"));
    let loader = ParameterLoader::<B>::new(&fixture.file, context.as_ref())
        .unwrap_or_else(|_| panic!("expert gather fixture header rejected"));
    let spec: AnyWeightMatrixSpec = serde_json::from_value(serde_json::json!({
        "type":"MLXSpec","bits":4,"group_size":GROUP_SIZE,"layout":"output_input"
    }))
    .unwrap();
    let bank = WeightMatrix::<B>::load(
        &loader.tree().subtree("bank"),
        spec,
        Layout::OutputInput,
        BANK_ROWS as u32,
        REDUCTION as u32,
        DataType::BF16,
    )
    .unwrap_or_else(|_| panic!("standard expert bank load failed"));
    loader.tree().assert_all_tensors_validated().unwrap();
    let mut input = context.create_buffer(fixture.input.len() * 2).unwrap();
    input.copyin(&fixture.input);
    let mut gather = context.create_buffer(fixture.gather.len() * 4).unwrap();
    gather.copyin(&fixture.gather);
    assert!(8 % TOP_K != 0 && INPUT_ROWS % 8 != 0, "fixture no longer crosses routing/chunk boundaries");
    for output_dtype in [DataType::F32, DataType::BF16] {
        let element_bytes = output_dtype.size_in_bytes();
        let mut output = context.create_buffer(INPUT_ROWS * OUTPUTS * element_bytes).unwrap();
        match output_dtype {
            DataType::F32 => output.copyin(&vec![f32::NAN; INPUT_ROWS * OUTPUTS]),
            DataType::BF16 => output.copyin(&vec![bf16::NAN; INPUT_ROWS * OUTPUTS]),
            _ => unreachable!(),
        }
        let mut kernel =
            <B::Kernels as Kernels>::MatmulKernel::new(context.as_ref(), DataType::BF16, DataType::BF16, output_dtype)
                .unwrap_or_else(|_| panic!("standard gathered kernel unavailable"));
        let mut commands = context.create_command_buffer(None, None).unwrap();
        for first in (0..INPUT_ROWS).step_by(8) {
            let count = (INPUT_ROWS - first).min(8);
            kernel
                .encode(
                    MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values: (&input).subrange(first * REDUCTION * 2..(first + count) * REDUCTION * 2),
                            offset: 0,
                        },
                        // Keep the entire bank: GroupOutput stride is 68,
                        // whereas the gathered output row width is only 13.
                        b: bank.matmul_b(),
                        b_leading_dimension: None,
                        b_transpose: true,
                        d: (&mut output)
                            .subrange_mut(first * OUTPUTS * element_bytes..(first + count) * OUTPUTS * element_bytes),
                        d_transform: MatmulDOps::none(),
                        gather_indices: Some((&gather).subrange(first * OUTPUTS * 4..(first + count) * OUTPUTS * 4)),
                        m: count as u32,
                        n: OUTPUTS as u32,
                        k: REDUCTION as u32,
                    },
                    &mut commands,
                )
                .unwrap_or_else(|_| panic!("standard expert gather encoding failed"));
        }
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let actual: Vec<f32> = match output_dtype {
            DataType::F32 => (&output).as_slice::<f32>().to_vec(),
            DataType::BF16 => (&output).as_slice::<bf16>().iter().map(|value| value.to_f32()).collect(),
            _ => unreachable!(),
        };
        for input_row in 0..INPUT_ROWS {
            for column in 0..OUTPUTS {
                let value = f64::from(actual[input_row * OUTPUTS + column]);
                let (expected, absolute_terms) = fixture.scalar(input_row, column);
                // Dyadic BF16 inputs/parameters and 4-bit codes keep individual
                // products exactly representable in F32. This conservative
                // gamma-style bound covers 256-term accumulation/reduction;
                // BF16 output additionally permits one rounding-bin spacing.
                let accumulation = 4e-6 + (REDUCTION + 8) as f64 * f64::from(f32::EPSILON) * absolute_terms;
                let rounding = if output_dtype == DataType::BF16 {
                    let magnitude = bf16::from_f32(expected.abs() as f32);
                    f64::from(bf16::from_bits(magnitude.to_bits() + 1).to_f32() - magnitude.to_f32())
                } else {
                    0.0
                };
                assert!(
                    value.is_finite() && (value - expected).abs() <= accumulation + rounding,
                    "group64 affine expert gather diverged from independent scalar dot"
                );
            }
        }
    }
}

#[cfg(backend = "cpu")]
#[stark_test]
fn q4_group64_full_bank_expert_gather_matches_scalar_cpu() {
    check::<crate::backends::cpu::Cpu>();
}

#[cfg(backend = "metal")]
#[stark_test]
fn q4_group64_full_bank_expert_gather_matches_scalar_metal() {
    check::<crate::backends::metal::Metal>();
}
