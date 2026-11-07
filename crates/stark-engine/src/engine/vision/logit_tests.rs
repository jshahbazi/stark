//! Private full-model last-prompt-logit qualification. No production capture API.

use std::path::{Path, PathBuf};

use half::bf16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;

fn checked_bytes(
    root: &Path,
    definition: &Value,
) -> Vec<u8> {
    let path = definition["file"].as_str().expect("invalid private logit reference name");
    let data = std::fs::read(root.join(path)).unwrap_or_else(|_| panic!("private logit reference unavailable"));
    let hash = format!("{:x}", Sha256::digest(&data));
    assert!(definition["sha256"].as_str() == Some(hash.as_str()), "private logit reference hash mismatch");
    data
}

fn u32_values(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len().is_multiple_of(4), "invalid private token/position byte count");
    bytes.chunks_exact(4).map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())).collect()
}

fn affine_dot_bitplanes<B: Backend>() {
    use crate::{
        backends::common::{
            CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::{
                QwenAffineReferenceMatmulBF16Kernel, VisionCastBF16Kernel, VisionMatmulBF16Kernel,
                matmul::{QuantParams, QuantParamsLayout},
            },
        },
        tests::helpers::{buffer_to_vec, create_buffer_with_data},
    };
    let context = B::Context::new().unwrap();
    let (m, n, k) = (3_u32, 5_u32, 128_u32);
    for bits in [4_u32, 5, 6, 8] {
        for layout in [QuantParamsLayout::OutputGroup, QuantParamsLayout::GroupOutput] {
            let params = QuantParams::new(layout, n, k / 64);
            let strides = params.scale_strides();
            let scale_shape = params.scale_shape();
            let mut scales = vec![bf16::ZERO; (scale_shape[0] * scale_shape[1]) as usize];
            let mut biases = scales.clone();
            let row_bytes = (k * bits / 8) as usize;
            let mut packed = vec![0_u8; n as usize * row_bytes];
            let mut dense = vec![bf16::ZERO; (n * k) as usize];
            for row in 0..n {
                for inner in 0..k {
                    let group = row * strides.output_stride + inner / 64 * strides.group_stride;
                    let scale = bf16::from_f32(0.1 + row as f32 / 32.0 + (inner / 64) as f32 / 64.0);
                    let bias = bf16::from_f32(-0.2 - row as f32 / 16.0);
                    scales[group as usize] = scale;
                    biases[group as usize] = bias;
                    let code = (row * 19 + inner * 7 + inner / 11) % (1 << bits);
                    let bit = inner * bits;
                    let offset = row as usize * row_bytes + (bit / 8) as usize;
                    packed[offset] |= (code << (bit % 8)) as u8;
                    if bit % 8 + bits > 8 {
                        packed[offset + 1] |= (code >> (8 - bit % 8)) as u8;
                    }
                    dense[(row * k + inner) as usize] = bf16::from_f32(code as f32 * scale.to_f32() + bias.to_f32());
                }
            }
            let input = (0..m * k).map(|index| bf16::from_f32((index % 31) as f32 / 16.0 - 1.0)).collect::<Vec<_>>();
            let input = create_buffer_with_data::<B, bf16>(&context, &input);
            let codes = create_buffer_with_data::<B, u8>(&context, &packed);
            let scales = create_buffer_with_data::<B, bf16>(&context, &scales);
            let biases = create_buffer_with_data::<B, bf16>(&context, &biases);
            let dense = create_buffer_with_data::<B, bf16>(&context, &dense);
            let mut actual = context.create_buffer((m * n * 2) as usize).unwrap();
            let mut expected = context.create_buffer((m * n * 2) as usize).unwrap();
            let mut accumulated = context.create_buffer((m * n * 4) as usize).unwrap();
            let packed_dot = <B::Kernels as Kernels>::QwenAffineReferenceMatmulBF16Kernel::new(&context).unwrap();
            let dense_dot = <B::Kernels as Kernels>::VisionMatmulBF16Kernel::new(&context).unwrap();
            let cast = <B::Kernels as Kernels>::VisionCastBF16Kernel::new(&context).unwrap();
            let mut commands = context.create_command_buffer(None, None).unwrap();
            packed_dot.encode(
                &input,
                &codes,
                &scales,
                &biases,
                &mut actual,
                m,
                n,
                k,
                bits,
                strides.output_stride,
                strides.group_stride,
                &mut commands,
            );
            dense_dot.encode(&input, &dense, &mut accumulated, m, n, k, k, 1, 32, &mut commands);
            cast.encode(&accumulated, &mut expected, m * n, &mut commands);
            commands.end_encoding().submit().wait_until_completed().unwrap();
            assert_eq!(buffer_to_vec::<B, bf16>(&actual), buffer_to_vec::<B, bf16>(&expected));
        }
    }
}

#[stark_test]
fn qwen_affine_dot_decodes_group64_bitplanes_before_contraction() {
    crate::tests::helpers::for_each_backend!(affine_dot_bitplanes);
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn qwen35_affine_dot_witness_last_prompt_logit_reference_metal() {
    use crate::backends::{
        common::{Kernels, kernel::matmul::MatmulKernel},
        metal::Metal,
    };
    let path = PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private exact artifact path"));
    let mut weights = std::fs::File::open(path.join("model.safetensors")).expect("private witness weights unavailable");
    let hash = file_digest(&mut weights).expect("private witness artifact hashing failed");
    assert_eq!(
        hash, "7e127dfb8c78f22b94edd35e52717f50e3d70398fe3668abd2290d088b5b1343",
        "contraction witness admits only the frozen small artifact"
    );
    <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::with_reference_dot_witness(|| {
        reference::<Metal>("metal-affine-dot-witness");
    });
}

fn reference<B: Backend>(backend: &str) {
    let path = PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private exact artifact path"));
    let root = PathBuf::from(
        std::env::var_os("STARK_VISION_LOGIT_REFERENCE_PATH").expect("set private full-model logit reference path"),
    );
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(root.join("manifest.json")).unwrap_or_else(|_| panic!("private logit manifest unavailable")),
    )
    .unwrap_or_else(|_| panic!("invalid private logit manifest"));
    let capacity = manifest["context_capacity"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .expect("invalid private logit context capacity");
    let engine = Engine::<B>::new().unwrap_or_else(|_| panic!("private logit engine setup failed"));
    let mut model = engine
        .load_vision_language_model(&path)
        .unwrap_or_else(|_| panic!("private logit exact artifact loading failed"));
    let vocab = model.vocab_size() as usize;
    assert!(manifest["vocab_size"].as_u64() == Some(vocab as u64), "unexpected private logit vocabulary");
    if manifest["schema_version"] == 2 {
        // Dynamic-family replays must bind the exact admitted artifact and
        // frozen execution controls. A vector from a different quantization,
        // template or source revision cannot qualify this loaded instance.
        let identity = model.artifact_identity();
        let expected = json!({"artifact_schema_version":identity.schema_version,
            "config_sha256":identity.config_sha256,"artifact_sha256":identity.artifact_sha256,
            "tokenizer_sha256":identity.tokenizer_sha256,"processor_sha256":identity.processor_sha256,
            "template_sha256":identity.template_sha256});
        assert!(manifest["artifact_identity"] == expected, "private logit artifact identity differs");
        let (repo, revision) = model.source_identity();
        assert!(manifest["source"] == json!({"repo":repo,"revision":revision}), "private logit source differs");
        let (profile, version) = model.family_profile();
        assert!(manifest["family"] == json!({"profile":profile,"version":version}), "private logit family differs");
        assert!(
            manifest["sampling"] == json!({"mode":"greedy"}) && manifest["mtp"] == false,
            "private logit execution controls differ"
        );
    } else {
        // Preserve the original immutable 0.8B reference replay contract.
        assert!(manifest.get("schema_version").is_none(), "unsupported private logit manifest version");
        assert!(manifest["threads"].as_u64() == Some(8), "unexpected private reference thread count");
        assert!(vocab == 248320 && model.model_dim == 1024, "unexpected legacy private logit geometry");
    }
    let config: Value = serde_json::from_slice(
        &std::fs::read(path.join("config.json")).unwrap_or_else(|_| panic!("private model metadata unavailable")),
    )
    .unwrap_or_else(|_| panic!("invalid private model metadata"));
    let layer_count = config["decoder_config"]["transformer_config"]["layer_configs"]
        .as_array()
        .filter(|layers| !layers.is_empty() && layers.len() <= 256)
        .expect("invalid private layer count")
        .len();
    let output = root.join(format!("native-{backend}"));
    std::fs::create_dir_all(&output).unwrap_or_else(|_| panic!("private logit output setup failed"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&output, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let mut metrics = Vec::new();
    for (index, case) in manifest["cases"].as_array().expect("invalid private logit cases").iter().enumerate() {
        let folder = root.join(case["name"].as_str().expect("invalid private logit case"));
        let image_bytes = checked_bytes(&folder, &case["image"]);
        let unexpanded = u32_values(&checked_bytes(&folder, &case["files"]["unexpanded_tokens"]));
        let expected_tokens = u32_values(&checked_bytes(&folder, &case["files"]["tokens"]));
        let expected_positions = u32_values(&checked_bytes(&folder, &case["files"]["positions"]));
        let expected_logits = checked_bytes(&folder, &case["files"]["logits"]);
        assert!(expected_logits.len() == vocab * 4, "incomplete private full-vocabulary reference");
        let image = model.prepare_image(&image_bytes).unwrap_or_else(|_| panic!("private image preparation failed"));
        assert!(
            u64::from(image.image_token_count()) == case["image_tokens"].as_u64().unwrap(),
            "private image token count differs"
        );
        for (axis, size) in image.grid().iter().enumerate() {
            assert!(u64::from(*size) == case["grid"][axis].as_u64().unwrap(), "private processed grid differs");
        }
        let (tokens, start) = model
            .prepare_prompt(unexpanded.into_iter().map(u64::from).collect(), &image, capacity)
            .unwrap_or_else(|_| panic!("private image prompt preparation failed"));
        assert!(
            tokens.len() == expected_tokens.len()
                && tokens.iter().zip(&expected_tokens).all(|(a, e)| *a == u64::from(*e)),
            "private expanded image prompt differs"
        );
        assert!(start as u64 == case["token_start"].as_u64().unwrap(), "private image span differs");
        let input = model
            .preflight_prepared_prefill(tokens.clone(), &image, start, capacity)
            .unwrap_or_else(|_| panic!("private image preflight failed"));
        assert!(
            input.positions.len() * 3 == expected_positions.len()
                && input.positions.iter().flatten().copied().eq(expected_positions),
            "private full-model M-RoPE positions differ"
        );
        assert!(
            input.position_delta == case["position_delta"].as_i64().unwrap(),
            "private full-model M-RoPE delta differs"
        );
        drop(input);
        let mut state =
            model.create_empty_state(Some(capacity), 0).unwrap_or_else(|_| panic!("private logit state setup failed"));
        state.vision_prefill_logits_readback = Some(
            engine
                .context
                .create_buffer(vocab * size_of::<bf16>())
                .unwrap_or_else(|_| panic!("private logit readback allocation failed")),
        );
        let input_len = tokens.len();
        if std::env::var_os("STARK_VISION_DECODER_LAYER_TRACE").is_some() {
            // Admitted model width and bounded layer count. These buffers
            // remain private and are read only after stream Drop drains.
            let row_bytes = model.model_dim as usize * size_of::<bf16>();
            let trace_bytes = input_len
                .checked_mul(row_bytes)
                .and_then(|n| n.checked_mul(layer_count))
                .expect("private layer trace size overflow");
            assert!(trace_bytes <= 512 * 1024 * 1024, "private layer trace exceeds bounded budget");
            let buffers = (0..layer_count)
                .map(|_| engine.context.create_buffer(input_len * row_bytes).unwrap())
                .collect::<Box<[_]>>();
            state.vision_prefill_layer_readbacks = Some((row_bytes, buffers));
        }
        let mut options = model.default_stream_options();
        options.sampling_method = crate::engine::language_model::stream::SamplingMethod::Greedy;
        {
            let stream = model
                .stream_prepared_image(tokens, image, start, capacity, &mut state, options)
                .unwrap_or_else(|_| panic!("private image prefill failed"));
            assert!(stream.metrics().num_tokens_prefilled == input_len, "private prefill count differs");
            assert!(stream.metrics().num_decode_forward_passes == 0, "private logit capture entered decode");
            drop(stream);
        }
        assert!(state.tokens().len() == input_len, "private logit capture consumed an extra token");
        assert!(!model.is_fenced(), "private image prefill fenced the instance");
        if let Some((_, readbacks)) = state.vision_prefill_layer_readbacks.take() {
            for (layer, readback) in readbacks.iter().enumerate() {
                let values = readback.as_slice::<bf16>();
                let raw = values.iter().flat_map(|value| value.to_bits().to_le_bytes()).collect::<Vec<_>>();
                std::fs::write(output.join(format!("case_{index}_layer_{layer}.bf16")), raw)
                    .unwrap_or_else(|_| panic!("private layer trace write failed"));
            }
        }
        let readback = state.vision_prefill_logits_readback.take().expect("private logit readback missing");
        let actual = (&readback).as_slice::<bf16>();
        let (mut squared_error, mut squared_actual, mut squared_expected, mut product) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut actual_top, mut expected_top, mut actual_max, mut expected_max) =
            (0usize, 0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let (mut maximum_error, mut mismatches) = (0.0f64, 0usize);
        let mut raw = Vec::with_capacity(actual.len() * 4);
        for (token, (actual, expected)) in actual.iter().zip(expected_logits.chunks_exact(4)).enumerate() {
            let a = actual.to_f32();
            let e = f32::from_le_bytes(expected.try_into().unwrap());
            assert!(a.is_finite() && e.is_finite(), "nonfinite private last-prompt logits");
            if a > actual_max {
                actual_max = a;
                actual_top = token;
            }
            if e > expected_max {
                expected_max = e;
                expected_top = token;
            }
            let difference = f64::from(a) - f64::from(e);
            squared_error += difference * difference;
            squared_actual += f64::from(a) * f64::from(a);
            squared_expected += f64::from(e) * f64::from(e);
            product += f64::from(a) * f64::from(e);
            maximum_error = maximum_error.max(difference.abs());
            mismatches += usize::from(a.to_bits() != e.to_bits());
            raw.extend_from_slice(&a.to_le_bytes());
        }
        std::fs::write(output.join(format!("case_{index}_logits.f32")), raw)
            .unwrap_or_else(|_| panic!("private native logit write failed"));
        metrics.push(json!({"case_index":index,"backend":backend,"input_tokens":input_len,"vocabulary":actual.len(),"relative_rms_error":(squared_error/squared_expected).sqrt(),"cosine_similarity":product/(squared_actual*squared_expected).sqrt(),"top1_matches":actual_top==expected_top,"bf16_mismatches":mismatches,"maximum_absolute_error":maximum_error}));
    }
    println!("{}", json!({"last_prompt_logit_metrics":metrics}));
    std::fs::write(
        root.join(format!("native-logit-{backend}-metrics.json")),
        serde_json::to_vec_pretty(&metrics).unwrap(),
    )
    .unwrap_or_else(|_| panic!("private logit aggregate write failed"));
    assert!(
        metrics.iter().all(|case| case["relative_rms_error"].as_f64().unwrap() <= 0.02
            && case["cosine_similarity"].as_f64().unwrap() >= 0.999
            && case["top1_matches"].as_bool() == Some(true)),
        "private last-prompt logit gate failed"
    );
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn qwen35_vision_last_prompt_logit_reference_cpu() {
    reference::<crate::backends::cpu::Cpu>("cpu");
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn qwen35_vision_last_prompt_logit_reference_metal() {
    reference::<crate::backends::metal::Metal>("metal");
}

/// Same-input projection controls isolate packed matrix arithmetic without
/// allocating a complete decoder. They are diagnostics, not a parity gate.
#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn qwen35_affine_projection_reference_metal() {
    use crate::{
        backends::{
            common::{
                BufferMut, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
                kernel::{QwenAffineReferenceMatmulBF16Kernel, matmul::MatmulB},
            },
            metal::Metal,
        },
        config::weight_matrix::Layout,
        encodable_block::{
            linear::{Gather, LinearMatmul},
            weight_matrix::WeightMatrix,
        },
        parameters::{NumericalProfile, ParameterLoader},
    };
    let artifact = PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private exact artifact path"));
    let root =
        PathBuf::from(std::env::var_os("STARK_QWEN_AFFINE_REFERENCE_PATH").expect("set private affine reference path"));
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(root.join("manifest.json")).unwrap_or_else(|_| panic!("private affine manifest unavailable")),
    )
    .unwrap_or_else(|_| panic!("invalid private affine manifest"));
    for (name, _) in ARTIFACT_FILES {
        let key = match name {
            "config.json" => "config_sha256",
            "model.safetensors" => "artifact_sha256",
            "tokenizer.json" => "tokenizer_sha256",
            "preprocessor_config.json" => "processor_sha256",
            "chat_template.jinja" => "template_sha256",
            _ => unreachable!(),
        };
        let mut file =
            std::fs::File::open(artifact.join(name)).unwrap_or_else(|_| panic!("private affine artifact unavailable"));
        let hash = file_digest(&mut file).unwrap_or_else(|_| panic!("private affine artifact hashing failed"));
        assert!(
            manifest["artifact_identity"][key].as_str() == Some(hash.as_str()),
            "private affine artifact identity differs"
        );
    }
    let context = <Metal as Backend>::Context::new().unwrap_or_else(|_| panic!("private affine context setup failed"));
    let file = std::fs::File::open(artifact.join("model.safetensors"))
        .unwrap_or_else(|_| panic!("private affine weights unavailable"));
    let loader = ParameterLoader::<Metal>::new(&file, &context)
        .unwrap_or_else(|_| panic!("private affine parameter setup failed"))
        .with_numerical_profile(NumericalProfile::Qwen35AffineBF16);
    let mut metrics = Vec::new();
    let exact_dot = std::env::var_os("STARK_QWEN_AFFINE_DOT_WITNESS").is_some();
    for (case_index, case) in manifest["cases"].as_array().expect("invalid private affine cases").iter().enumerate() {
        let folder = root.join(case["name"].as_str().expect("invalid private affine case name"));
        for stage in case["projection_stages"].as_array().expect("missing private affine projection controls") {
            let input = checked_bytes(&folder, &stage["input"]);
            let expected = checked_bytes(&folder, &stage["expected"]);
            let integer = |value: &Value| {
                value
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .filter(|n| *n > 0)
                    .expect("invalid private affine dimension")
            };
            let m = integer(&stage["input"]["shape"][0]);
            let k = integer(&stage["input"]["shape"][1]);
            let n = integer(&stage["expected"]["shape"][1]);
            assert!(m <= 128 && k <= 17408 && n <= 34816, "private affine projection exceeds bounded diagnostic");
            assert!(
                input.len() == m as usize * k as usize * 2 && expected.len() == m as usize * n as usize * 2,
                "private affine shape differs"
            );
            let tree = loader.tree().subtree(stage["tree"].as_str().expect("invalid private affine parameter tree"));
            let reference_weights = exact_dot.then(|| {
                WeightMatrix::<Metal>::load(
                    &tree,
                    tree.metadata("spec").unwrap_or_else(|_| panic!("private affine witness metadata unavailable")),
                    Layout::OutputInput,
                    n,
                    k,
                    DataType::BF16,
                )
                .unwrap_or_else(|_| panic!("private affine witness weights unavailable"))
            });
            let matrix = (!exact_dot).then(|| {
                LinearMatmul::<Metal>::load(
                    &context,
                    tree.metadata("spec").unwrap_or_else(|_| panic!("private affine matrix metadata unavailable")),
                    k,
                    n,
                    DataType::BF16,
                    DataType::BF16,
                    DataType::BF16,
                    &tree,
                    None,
                    None,
                )
                .unwrap_or_else(|_| panic!("private affine matrix loading failed"))
            });
            let mut values =
                context.create_buffer(input.len()).unwrap_or_else(|_| panic!("private affine input allocation failed"));
            (&mut values).copyin(&input);
            let mut commands = context
                .create_command_buffer(None, None)
                .unwrap_or_else(|_| panic!("private affine commands unavailable"));
            let output = if let Some(weights) = &reference_weights {
                let MatmulB::Quantized(quantized) = weights.matmul_b() else {
                    panic!("private affine witness requires packed weights");
                };
                assert!(
                    quantized.group_size == 64
                        && !quantized.signed_codes
                        && matches!(quantized.mode.bits(), 4 | 5 | 6 | 8)
                );
                let biases = quantized.biases().expect("private affine witness requires affine biases");
                let strides = quantized.params.scale_strides();
                let mut output = commands.allocate_scratch_for_shape(&[m, n], DataType::BF16).unwrap();
                let dot = <<Metal as Backend>::Kernels as Kernels>::QwenAffineReferenceMatmulBF16Kernel::new(&context)
                    .unwrap_or_else(|_| panic!("private packed dot witness setup failed"));
                dot.encode(
                    &values,
                    quantized.codes,
                    quantized.scales,
                    biases,
                    &mut output,
                    m,
                    n,
                    k,
                    quantized.mode.bits(),
                    strides.output_stride,
                    strides.group_stride,
                    &mut commands,
                );
                output
            } else {
                matrix
                    .as_ref()
                    .unwrap()
                    .encode_with_a(
                        crate::backends::common::kernel::matmul::MatmulA::FullPrecision {
                            values: &values,
                            offset: 0,
                        },
                        m,
                        None::<Gather<&<Metal as Backend>::GlobalBuffer>>,
                        &mut commands,
                    )
                    .unwrap_or_else(|_| panic!("private affine encoding failed"))
            };
            let mut readback =
                context.create_buffer(expected.len()).unwrap_or_else(|_| panic!("private affine readback unavailable"));
            commands.encode_copy(&output, &mut readback);
            commands
                .end_encoding()
                .submit()
                .wait_until_completed()
                .unwrap_or_else(|_| panic!("private affine execution failed"));
            let actual = (&readback).as_slice::<bf16>();
            let (mut error, mut square, mut mismatches) = (0.0f64, 0.0f64, 0usize);
            for (actual, expected) in actual.iter().zip(expected.chunks_exact(2)) {
                let expected = bf16::from_bits(u16::from_le_bytes(expected.try_into().unwrap()));
                let (a, e) = (f64::from(actual.to_f32()), f64::from(expected.to_f32()));
                assert!(a.is_finite() && e.is_finite(), "nonfinite private affine projection");
                error += (a - e) * (a - e);
                square += e * e;
                mismatches += usize::from(actual.to_bits() != expected.to_bits());
            }
            metrics.push(json!({"case_index":case_index,"stage":stage["name"],"elements":actual.len(),"exact_dot_witness":exact_dot,
                "relative_rms":(error/square).sqrt(),"bf16_mismatches":mismatches}));
        }
    }
    println!("{}", json!({"private_affine_projection_metrics":metrics}));
}

/// Reproduce the worker's first text check without printing request content,
/// token IDs, generated output or backend exception strings.
#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn qwen35_vision_worker_text_diagnostic_metal() {
    use crate::{
        backends::metal::{Metal, MetalError},
        encodable_block::decoder::DecoderError,
    };
    fn failure(error: NativeVisionError<Metal>) -> ! {
        let mut result = json!({"kind":format!("{:?}",error.kind())});
        let backend = match &error.failure {
            NativeVisionFailure::Stream(LanguageModelStreamError::Backend(error)) => {
                result["stream_stage"] = json!("backend");
                Some(error)
            },
            NativeVisionFailure::Stream(LanguageModelStreamError::Decoder(error)) => {
                result["stream_stage"] = json!("decoder");
                match error {
                    DecoderError::Backend(error) => Some(error),
                    DecoderError::InvalidInput(error) => {
                        result["static_input_sha256"] = json!(format!("{:x}", Sha256::digest(error.0.as_bytes())));
                        None
                    },
                    _ => None,
                }
            },
            NativeVisionFailure::Stream(LanguageModelStreamError::InvalidMultimodal) => {
                result["stream_stage"] = json!("multimodal_preflight");
                None
            },
            NativeVisionFailure::Backend(error) => Some(error),
            _ => None,
        };
        if let Some(error) = backend {
            let stage = match error {
                MetalError::InvalidInput(error) => {
                    result["static_input_sha256"] = json!(format!("{:x}", Sha256::digest(error.0.as_bytes())));
                    "invalid_input"
                },
                MetalError::CannotCreateBuffer | MetalError::CannotCreateHeap => "allocation",
                MetalError::CannotCreatePipelineState {
                    ..
                }
                | MetalError::CannotCreateFunction(_) => "pipeline",
                MetalError::KernelDispatchFailed(_) => "dispatch",
                MetalError::CommandBufferExecution(_) | MetalError::CommandBufferWait(_) => "execution",
                _ => "setup",
            };
            result["backend_stage"] = json!(stage);
        }
        println!("{}", json!({"qwen_text_diagnostic_failure":result}));
        panic!("private Qwen text diagnostic failed");
    }
    let path = PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private exact artifact path"));
    let engine = Engine::<Metal>::new().unwrap_or_else(|_| panic!("private diagnostic engine setup failed"));
    let mut model = engine.load_vision_language_model(&path).unwrap_or_else(|error| failure(error));
    println!("{}", json!({"qwen_text_diagnostic_stage":"loaded"}));
    let mut state = model.create_empty_state(Some(2048), 0).unwrap_or_else(|error| failure(error));
    println!("{}", json!({"qwen_text_diagnostic_stage":"state_created"}));
    let tokens = model.tokenizer().encode("Hello.", false).expect("private diagnostic tokenization failed");
    // Match the worker check prefill shape without logging token IDs.
    let tokens = tokens.get_ids().iter().copied().cycle().take(14).map(u64::from).collect::<Vec<_>>();
    let mut options = model.default_stream_options();
    options.sampling_method = crate::engine::language_model::stream::SamplingMethod::Greedy;
    let mut stream = model.stream(&tokens, &mut state, options).unwrap_or_else(|error| failure(error));
    println!("{}", json!({"qwen_text_diagnostic_stage":"prefilled"}));
    for index in 0..8 {
        match stream.next() {
            Some(Ok(_)) => {},
            Some(Err(error)) => failure(error.into()),
            None => panic!("private diagnostic stream ended unexpectedly"),
        }
        println!("{}", json!({"qwen_text_diagnostic_stage":"decoded","completed_steps":index+1}));
    }
    drop(stream);
    assert!(!model.is_fenced(), "private diagnostic model fenced");
}
