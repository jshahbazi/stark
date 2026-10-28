//! Ignored, private same-input primitive replay. No production observer API.

use std::{fs::File, path::Path};

use half::bf16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    array::ArrayElement,
    backends::common::{CommandBufferExecutable, CommandBufferPending, gpu_types::trie::TrieNode},
    parameters::{NumericalProfile, ParameterLoader},
};

fn read(
    root: &Path,
    definition: &Value,
) -> Vec<u8> {
    let data = std::fs::read(root.join(definition["file"].as_str().expect("invalid private stage name")))
        .unwrap_or_else(|_| panic!("private DeltaNet stage unavailable"));
    assert!(
        format!("{:x}", Sha256::digest(&data)) == definition["sha256"].as_str().unwrap(),
        "private stage hash differs"
    );
    data
}

fn bfloat(data: &[u8]) -> Vec<bf16> {
    assert!(data.len().is_multiple_of(2));
    data.chunks_exact(2).map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap()))).collect()
}

fn floats(data: &[u8]) -> Vec<f32> {
    assert!(data.len().is_multiple_of(4));
    data.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect()
}

fn upload<B: Backend, T: ArrayElement>(
    values: &[T],
    commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
) -> Result<B::ScratchBuffer, B::Error> {
    let source = commands.allocate_constant_from_slice(values)?;
    let mut output = commands.allocate_scratch(std::mem::size_of_val(values))?;
    commands.encode_copy(&source, &mut output);
    Ok(output)
}

fn capture<B: Backend>(
    name: &str,
    data_type: DataType,
    values: &B::ScratchBuffer,
    context: &B::Context,
    commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    output: &mut Vec<(String, DataType, B::GlobalBuffer)>,
) -> Result<(), B::Error> {
    let mut target = context.create_buffer(values.size())?;
    commands.encode_copy(values, &mut target);
    output.push((name.into(), data_type, target));
    Ok(())
}

fn same_input<B: Backend>(backend: &str) {
    let artifact =
        std::path::PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set exact private artifact"));
    let root = std::path::PathBuf::from(
        std::env::var_os("STARK_DELTA_REFERENCE_PATH").expect("set private DeltaNet references"),
    );
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).expect("private manifest unavailable"))
            .expect("invalid private manifest");
    assert_eq!(manifest["threads"].as_u64(), Some(8));
    let config: Value =
        serde_json::from_slice(&std::fs::read(artifact.join("config.json")).expect("private config unavailable"))
            .expect("invalid private config");
    let transformer = &config["decoder_config"]["transformer_config"];
    let delta_config: DeltaNetConfig = serde_json::from_value(transformer["layer_configs"][0]["mixer_config"].clone())
        .expect("invalid private DeltaNet config");
    assert!(delta_config.reference_bf16);
    let context = B::Context::new().unwrap_or_else(|_| panic!("private context failed"));
    let file = File::open(artifact.join("model.safetensors")).expect("private weights unavailable");
    let reference_profile = std::env::var_os("STARK_DELTA_REFERENCE_PROFILE").is_some();
    let loader = ParameterLoader::<B>::new(&file, &context)
        .unwrap_or_else(|_| panic!("private weights indexing failed"))
        .with_numerical_profile(if reference_profile {
            NumericalProfile::Qwen35ReferenceBF16
        } else {
            NumericalProfile::Legacy
        });
    let tree = loader.tree().subtree("decoder.transformer.layers.0.mixer");
    let (delta, _retained) = DeltaNet::<B>::new(
        transformer["model_dim"].as_u64().unwrap() as u32,
        DataType::BF16,
        &delta_config,
        &tree,
        &context,
    )
    .unwrap_or_else(|_| panic!("private DeltaNet loading failed"));
    let output = root.join(format!(
        "native-{backend}{}",
        if reference_profile {
            "-reference"
        } else {
            ""
        }
    ));
    std::fs::create_dir_all(&output).expect("private output setup failed");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&output, std::fs::Permissions::from_mode(0o700))
            .expect("private output permissions failed");
    }
    let mut metrics = Vec::new();
    for (index, case) in manifest["cases"].as_array().unwrap().iter().enumerate() {
        let rows = case["rows"].as_u64().unwrap() as u32;
        let refs = &case["files"];
        let hidden = bfloat(&read(&root, &refs["input"]));
        let projected = bfloat(&read(&root, &refs["projection"]));
        let convolved = bfloat(&read(&root, &refs["convolution"]));
        let mut full_state = delta.create_empty_state(None, &context).unwrap();
        let nodes = (0..rows)
            .map(|height| TrieNode {
                height,
                trie_start: height,
                trie_end: rows - 1,
            })
            .collect::<Vec<_>>();
        let batch = BatchTopology::new(&nodes, true);
        let mut commands =
            context.create_command_buffer(None, None).unwrap_or_else(|_| panic!("private commands failed"));
        let mut readbacks = Vec::new();
        let full_input = upload::<B, _>(&hidden, &mut commands).unwrap();
        let full_output =
            delta.encode(full_input, None, &batch, Some(MaybeMut::Mut(full_state.as_mut())), &mut commands).unwrap();
        capture::<B>("propagated_output", DataType::BF16, &full_output, &context, &mut commands, &mut readbacks)
            .unwrap();
        let native_projection =
            delta.in_projection.encode(upload::<B, _>(&hidden, &mut commands).unwrap(), rows, &mut commands).unwrap();
        capture::<B>("projection", DataType::BF16, &native_projection, &context, &mut commands, &mut readbacks)
            .unwrap();

        // Each following primitive starts from its own exact HF input; it never
        // consumes an upstream native result whose errors could propagate.
        let mut conv = upload::<B, _>(&projected, &mut commands).unwrap();
        let mut history =
            commands.allocate_scratch_for_shape(&[delta.conv_dim, delta.kernel_size - 1], DataType::F32).unwrap();
        commands.encode_fill(&mut history, 0);
        let mut state_out = commands.allocate_scratch(history.size()).unwrap();
        let mut padded = commands
            .allocate_scratch_for_shape(&[rows + delta.kernel_size - 1, delta.total_proj_dim], DataType::F32)
            .unwrap();
        if let Some(reference_conv) = &delta.reference_conv {
            let mut convolved_native = commands.allocate_scratch(conv.size()).unwrap();
            reference_conv.encode(
                &conv,
                &history,
                &delta.conv_weight,
                &mut convolved_native,
                &mut state_out,
                rows,
                delta.kernel_size,
                delta.total_proj_dim,
                delta.total_proj_dim,
                delta.kernel_size - 1,
                delta.conv_dim,
                &mut commands,
            );
            conv = convolved_native;
        } else {
            delta.conv_pack.encode(
                &history,
                &conv,
                &mut padded,
                delta.kernel_size - 1,
                delta.total_proj_dim,
                rows,
                delta.conv_dim,
                &mut commands,
            );
            delta.conv_scan.encode(
                &padded,
                &delta.conv_weight,
                delta.conv_bias.as_ref(),
                &mut conv,
                &mut state_out,
                rows,
                delta.kernel_size,
                delta.total_proj_dim,
                delta.kernel_size - 1,
                delta.conv_dim,
                delta.total_proj_dim,
                1,
                &mut commands,
            );
        }
        capture::<B>("convolution", DataType::BF16, &conv, &context, &mut commands, &mut readbacks).unwrap();
        let exact_conv = upload::<B, _>(&convolved, &mut commands).unwrap();
        let mut q = commands.allocate_scratch_for_shape(&[rows, delta.key_dim], DataType::F32).unwrap();
        let mut k = commands.allocate_scratch(q.size()).unwrap();
        let mut beta = commands.allocate_scratch_for_shape(&[rows, delta.num_heads], DataType::F32).unwrap();
        let mut decay = commands.allocate_scratch(beta.size()).unwrap();
        delta.delta_net_prefill_prep.encode(
            &exact_conv,
            &exact_conv,
            &delta.a_log,
            &delta.dt_bias,
            &mut q,
            &mut k,
            None::<&mut B::ScratchBuffer>,
            &mut beta,
            &mut decay,
            delta.num_heads,
            delta.num_groups,
            delta.key_dim,
            delta.value_dim,
            rows,
            delta.total_proj_dim,
            1,
            &mut commands,
        );
        if let Some(reference_decay) = &delta.reference_decay {
            reference_decay.encode(
                &exact_conv,
                &delta.a_log,
                &delta.dt_bias,
                &mut decay,
                rows,
                delta.num_heads,
                delta.conv_dim,
                delta.value_dim,
                delta.total_proj_dim,
                0,
                &mut commands,
            );
            let mut native_g = commands.allocate_scratch(decay.size()).unwrap();
            reference_decay.encode(
                &exact_conv,
                &delta.a_log,
                &delta.dt_bias,
                &mut native_g,
                rows,
                delta.num_heads,
                delta.conv_dim,
                delta.value_dim,
                delta.total_proj_dim,
                1,
                &mut commands,
            );
            capture::<B>("g", DataType::F32, &native_g, &context, &mut commands, &mut readbacks).unwrap();
        }
        if let Some(reference_l2) = &delta.reference_l2 {
            reference_l2.encode(&exact_conv, &mut q, &mut k, delta.key_dim, rows, delta.total_proj_dim, &mut commands);
        }
        for (name, value) in [("qnorm", &q), ("knorm", &k), ("beta", &beta), ("decay", &decay)] {
            capture::<B>(name, DataType::F32, value, &context, &mut commands, &mut readbacks).unwrap();
        }
        let q = upload::<B, _>(&floats(&read(&root, &refs["qnorm"])), &mut commands).unwrap();
        let k = upload::<B, _>(&floats(&read(&root, &refs["knorm"])), &mut commands).unwrap();
        let beta = upload::<B, _>(&floats(&read(&root, &refs["beta"])), &mut commands).unwrap();
        let decay = upload::<B, _>(&floats(&read(&root, &refs["decay"])), &mut commands).unwrap();
        let mut state = context
            .create_buffer(size_for_shape(&[delta.num_heads, delta.value_head_dim, delta.head_dim], DataType::F32))
            .unwrap();
        commands.encode_fill(&mut state, 0);
        let mut core = commands.allocate_scratch_for_shape(&[rows, delta.value_dim], DataType::BF16).unwrap();
        if let Some(reference_chunk) = &delta.reference_chunk {
            let log_decay = upload::<B, _>(&floats(&read(&root, &refs["g"])), &mut commands).unwrap();
            reference_chunk
                .encode(
                    &q,
                    &k,
                    &beta,
                    &log_decay,
                    &exact_conv,
                    &mut state,
                    &mut core,
                    rows,
                    delta.num_heads,
                    delta.num_groups,
                    delta.key_dim,
                    delta.value_dim,
                    delta.total_proj_dim,
                    &mut commands,
                )
                .unwrap();
        } else {
            delta.delta_net_prefill.encode(
                &q,
                &k,
                &beta,
                &decay,
                &exact_conv,
                &mut state,
                &mut core,
                delta.num_heads,
                delta.num_groups,
                delta.value_head_dim,
                delta.key_dim,
                delta.value_dim,
                rows,
                delta.value_head_dim.div_ceil(16),
                delta.total_proj_dim,
                &mut commands,
            );
        }
        capture::<B>("core", DataType::BF16, &core, &context, &mut commands, &mut readbacks).unwrap();
        if delta.reference_chunk.is_some() {
            let mut state_readback = context.create_buffer((&state).size()).unwrap();
            commands.encode_copy(&state, &mut state_readback);
            readbacks.push(("state".into(), DataType::F32, state_readback));
        }
        let mut norm = upload::<B, _>(&bfloat(&read(&root, &refs["core"])), &mut commands).unwrap();
        if let Some(reference_norm_gate) = &delta.reference_norm_gate {
            reference_norm_gate.encode(
                &mut norm,
                &exact_conv,
                &delta.norm_weight,
                rows,
                delta.num_heads,
                delta.value_dim,
                delta.conv_dim,
                delta.total_proj_dim,
                delta.norm_epsilon,
                &mut commands,
            );
        } else {
            delta.delta_net_norm_gate.encode(
                &mut norm,
                &exact_conv,
                &delta.norm_weight,
                delta.num_heads,
                delta.value_head_dim,
                delta.value_dim,
                delta.conv_dim,
                delta.total_proj_dim,
                delta.norm_epsilon,
                rows,
                1,
                &mut commands,
            );
        }
        capture::<B>("norm_gate", DataType::BF16, &norm, &context, &mut commands, &mut readbacks).unwrap();
        let exact_norm = upload::<B, _>(&bfloat(&read(&root, &refs["norm_gate"])), &mut commands).unwrap();
        let projected = delta.out_projection.encode(exact_norm, rows, &mut commands).unwrap();
        capture::<B>("output", DataType::BF16, &projected, &context, &mut commands, &mut readbacks).unwrap();
        commands
            .end_encoding()
            .submit()
            .wait_until_completed()
            .unwrap_or_else(|_| panic!("private primitive completion failed"));
        for (name, dtype, buffer) in readbacks {
            let (actual, bytes, extension) = if dtype == DataType::BF16 {
                let values = buffer.as_slice::<bf16>();
                (
                    values.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
                    values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect::<Vec<_>>(),
                    "bf16",
                )
            } else {
                let values = buffer.as_slice::<f32>();
                (values.to_vec(), values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), "f32")
            };
            let expected_name = if name == "propagated_output" {
                "output"
            } else {
                name.as_str()
            };
            let reference = read(&root, &refs[expected_name]);
            let expected = if dtype == DataType::BF16 {
                bfloat(&reference).into_iter().map(|v| v.to_f32()).collect()
            } else {
                floats(&reference)
            };
            assert!(
                actual.len() == expected.len() && actual.iter().all(|v| v.is_finite()),
                "private primitive shape/nonfinite result"
            );
            let (mut error, mut energy, mut mismatch, mut maximum) = (0.0, 0.0, 0usize, 0.0f64);
            for (a, b) in actual.into_iter().zip(expected) {
                let d = f64::from(a) - f64::from(b);
                error += d * d;
                energy += f64::from(b).powi(2);
                mismatch += usize::from(a != b);
                maximum = maximum.max(d.abs());
            }
            std::fs::write(output.join(format!("case_{index}_{name}.{extension}")), bytes)
                .expect("private primitive write failed");
            metrics.push(json!({"case":index,"stage":name,"relative_rms":(error/energy).sqrt(),"mismatches":mismatch,"maximum_error":maximum}));
        }
    }
    let metrics =
        json!({"backend":backend,"same_hf_input":true,"reference_profile":reference_profile,"stages":metrics});
    std::fs::write(output.join("aggregate.json"), serde_json::to_vec_pretty(&metrics).unwrap())
        .expect("private aggregate write failed");
    println!("{metrics}");
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore] // Requires exact private official artifact and HF stages.
fn delta_net_same_input_reference_cpu() {
    same_input::<crate::backends::cpu::Cpu>("cpu");
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore] // Requires exact private official artifact and HF stages.
fn delta_net_same_input_reference_metal() {
    same_input::<crate::backends::metal::Metal>("metal");
}
