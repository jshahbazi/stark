//! Explicit private OLMo preparation control; no fixtures or vectors enter CI.

use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use half::bf16;
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

#[path = "../../src/encodable_block/mixer/attention/qkv_norm.rs"]
mod production_qkv_norm;

use crate::{
    backends::{
        common::{
            Backend, Buffer, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            kernel::{AttentionPrepareKernel, Kernels},
        },
        metal::Metal,
    },
    config::{decoder::DecoderConfig, rope::AnyRoPEConfig, token_mixer::attention::AttentionConfig},
    data_type::DataType,
    encodable_block::{
        mixer::attention::rope::PrecalculatedRoPE,
        normalization::{Normalization, PostLayerScalar, ShortcutMode},
    },
    parameters::ParameterLoader,
};

fn private_output(path: &Path) -> Result<(), &'static str> {
    if path.exists() || path.symlink_metadata().is_ok() {
        return Err("output_exists");
    }
    let ignored = Command::new("git")
        .args(["check-ignore", "--quiet", "--"])
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|_| "output_ignore_check")?;
    if !ignored.success() {
        return Err("output_not_private");
    }
    Ok(())
}

fn prepared_inputs_probe() -> Result<(), &'static str> {
    let model = PathBuf::from(std::env::var_os("STARK_OLMO_TRACE_MODEL").ok_or("model_missing")?);
    let fixture = PathBuf::from(std::env::var_os("STARK_OLMO_PREPARED_FIXTURE").ok_or("fixture_missing")?);
    let output = PathBuf::from(std::env::var_os("STARK_OLMO_PREPARED_OUTPUT").ok_or("output_missing")?);
    let source_rope = std::env::var_os("STARK_OLMO_SOURCE_ROPE").is_some();
    let layer = std::env::var("STARK_OLMO_PREPARED_LAYER")
        .ok()
        .map(|x| x.parse::<usize>().map_err(|_| "layer_invalid"))
        .transpose()?
        .unwrap_or(0);
    if layer >= 32 {
        return Err("layer_unsupported");
    }
    private_output(&output)?;
    let config_bytes = fs::read(model.join("config.json")).map_err(|_| "config_read")?;
    let raw: serde_json::Value = serde_json::from_slice(&config_bytes).map_err(|_| "config_invalid")?;
    let config: DecoderConfig = serde_json::from_value(raw["decoder_config"].clone()).map_err(|_| "decoder_invalid")?;
    let attention: AttentionConfig = serde_json::from_value(
        raw["decoder_config"]["transformer_config"]["layer_configs"][layer]["mixer_config"].clone(),
    )
    .map_err(|_| "attention_invalid")?;
    let rope_config = config.transformer_config.layer_configs[layer].rope_config.as_ref().ok_or("rope_missing")?;
    if config.transformer_config.layer_configs.len() != 32
        || attention.num_heads != 32
        || attention.num_groups != 32
        || attention.head_dim != 128
        || !attention.qk_norm_full_width
        || !(matches!(rope_config, AnyRoPEConfig::UnscaledRoPEConfig(_) | AnyRoPEConfig::YARNRoPEConfig(_)))
    {
        return Err("unsupported_prepared_geometry");
    }
    let receipt_bytes = fs::read(fixture.join("receipt.json")).map_err(|_| "receipt_read")?;
    let receipt: serde_json::Value = serde_json::from_slice(&receipt_bytes).map_err(|_| "receipt_invalid")?;
    if receipt["device"] != "gpu"
        || receipt["mlx_version"] != "0.32.3"
        || receipt["mlx_lm_revision"] != "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"
    {
        return Err("source_identity_mismatch");
    }
    let rows = receipt["input_tokens"].as_u64().ok_or("source_rows")?;
    if !(1..=1024).contains(&rows) {
        return Err("source_rows");
    }
    let rows = rows as u32;
    let cached_step = receipt["cached_step"].as_bool().unwrap_or(false);
    let position_offset = receipt["position_offset"].as_u64().unwrap_or(0);
    if position_offset + u64::from(rows) > 1024 {
        return Err("position_bounds");
    }
    let position_offset = position_offset as u32;
    let q_width = attention.num_heads * attention.head_dim;
    let kv_width = attention.num_groups * attention.head_dim;
    let columns = q_width + 2 * kv_width;
    let read = |name: &str, shape: &[u32]| -> Result<Vec<f32>, &'static str> {
        let entry = receipt["components"]
            .as_array()
            .ok_or("components_missing")?
            .iter()
            .find(|entry| entry["component"] == name)
            .ok_or("component_missing")?;
        let expected_elements =
            shape.iter().try_fold(1usize, |n, &v| n.checked_mul(v as usize)).ok_or("shape_overflow")?;
        let bytes = fs::read(fixture.join("trace-prefill").join(format!("{name}.f32"))).map_err(|_| "tensor_read")?;
        if bytes.len() != expected_elements * 4
            || entry["elements"] != expected_elements
            || entry["shape"] != serde_json::json!(shape)
            || entry["sha256"] != format!("{:x}", Sha256::digest(&bytes))
        {
            return Err("tensor_identity");
        }
        let values: Vec<_> = bytes.as_chunks::<4>().0.iter().map(|x| f32::from_le_bytes(*x)).collect();
        if values.iter().any(|&x| !x.is_finite() || bf16::from_f32(x).to_f32().to_bits() != x.to_bits()) {
            return Err("source_not_bf16");
        }
        Ok(values)
    };
    let q = read(&format!("layer-{layer}-q-projection"), &[1, rows, q_width])?;
    let k = read(&format!("layer-{layer}-k-projection"), &[1, rows, kv_width])?;
    let v = read(&format!("layer-{layer}-v-projection"), &[1, rows, kv_width])?;
    let mut input = Vec::with_capacity(rows as usize * columns as usize);
    for row in 0..rows as usize {
        for (values, width) in [(&q, q_width), (&k, kv_width), (&v, kv_width)] {
            input.extend(values[row * width as usize..(row + 1) * width as usize].iter().map(|&x| bf16::from_f32(x)));
        }
    }
    let context = <Metal as Backend>::Context::new().map_err(|_| "context")?;
    let weights = File::open(model.join("model.safetensors")).map_err(|_| "weights_read")?;
    let loader = ParameterLoader::<Metal>::new(&weights, &context).map_err(|_| "loader")?;
    let norm = production_qkv_norm::QKVNorm::<Metal>::new(
        &context,
        DataType::BF16,
        attention.query_norm_config.clone(),
        attention.key_norm_config.clone(),
        attention.value_norm_config(),
        &loader.tree().subtree(&format!("decoder.transformer.layers.{layer}.mixer")),
        attention.num_heads,
        attention.num_groups,
        columns,
        attention.head_dim,
        attention.qk_norm_full_width,
    )
    .map_err(|_| "norm_new")?;
    let mut direct_norms = Vec::new();
    let direct_source_rms = std::env::var_os("STARK_OLMO_DIRECT_SOURCE_RMS").is_some();
    if direct_source_rms {
        for (name, norm_config) in
            [("query_norm", attention.query_norm_config.as_ref()), ("key_norm", attention.key_norm_config.as_ref())]
        {
            let mut config = norm_config.ok_or("direct_norm_config")?.clone();
            config.mlx_bf16_reduction = true;
            let width = if name == "query_norm" {
                q_width
            } else {
                kv_width
            };
            direct_norms.push(
                Normalization::<Metal>::new(
                    width,
                    None,
                    ShortcutMode::None,
                    PostLayerScalar::None,
                    DataType::BF16,
                    &config,
                    &loader.tree().subtree(&format!("decoder.transformer.layers.{layer}.mixer.{name}")),
                    &context,
                )
                .map_err(|_| "direct_norm_new")?,
            );
        }
    }
    let mut commands = context.create_command_buffer(None, None).map_err(|_| "commands")?;
    let constant = commands.allocate_constant_from_slice(&input).map_err(|_| "constant")?;
    let mut normalized =
        commands.allocate_scratch_for_shape(&[rows, columns], DataType::BF16).map_err(|_| "scratch")?;
    commands.encode_copy(&constant, &mut normalized);
    norm.encode(&mut normalized, rows, &mut commands).map_err(|_| "norm_encode")?;
    let rope = PrecalculatedRoPE::<Metal>::precalculate(
        rope_config,
        &(position_offset..position_offset + rows).collect::<Vec<_>>(),
        &mut commands,
    )
    .map_err(|_| "rope_tables")?;
    let source = if source_rope {
        Some(rope.source_parameters.ok_or("source_rope_metadata")?)
    } else {
        None
    };
    let prepare = <<Metal as Backend>::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        DataType::BF16,
        DataType::F32,
        true,
        true,
        attention.reference_bf16,
        source_rope,
        source.is_some_and(|p| p.proportional),
        source.is_some_and(|p| p.yarn),
    )
    .map_err(|_| "prepare_new")?;
    let mut queries = commands
        .allocate_scratch_for_shape(&[attention.num_heads, rows, attention.head_dim], DataType::BF16)
        .map_err(|_| "queries")?;
    let mut keys = commands.allocate_scratch_for_shape(&[rows, kv_width], DataType::BF16).map_err(|_| "keys")?;
    let mut values = commands.allocate_scratch_for_shape(&[rows, kv_width], DataType::BF16).map_err(|_| "values")?;
    prepare.encode(
        &normalized,
        &mut queries,
        Some(&mut keys),
        Some(&mut values),
        Some(&rope.cosines),
        Some(&rope.sines),
        source.and(rope.source_positions.as_ref()),
        source.map(|p| p.base),
        source.map(|p| p.rotary_dim),
        source.and(rope.source_yarn_parameters.as_ref()),
        attention.num_heads,
        Some(attention.num_groups),
        attention.head_dim,
        Some(rope.dim),
        Some(0),
        columns,
        rows,
        &mut commands,
    );
    let mut direct_constants = Vec::new();
    let mut direct_outputs = Vec::new();
    for (index, norm) in direct_norms.iter().enumerate() {
        let raw = if index == 0 {
            &q
        } else {
            &k
        };
        let input = raw.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>();
        let constant = commands.allocate_constant_from_slice(&input).map_err(|_| "direct_constant")?;
        let normalized = norm
            .encode(&constant, 0, rows, None::<&mut <Metal as Backend>::GlobalBuffer>, &mut commands)
            .map_err(|_| "direct_encode")?;
        direct_constants.push(constant);
        direct_outputs.push(normalized);
    }
    let mut copies = Vec::new();
    for buffer in [&normalized, &queries, &keys, &values] {
        let mut copy = context.create_buffer(buffer.size()).map_err(|_| "readback")?;
        commands.encode_copy(buffer, &mut copy);
        copies.push(copy);
    }
    for buffer in &direct_outputs {
        let mut copy = context.create_buffer(buffer.size()).map_err(|_| "direct_readback")?;
        commands.encode_copy(buffer, &mut copy);
        copies.push(copy);
    }
    drop(direct_constants);
    drop(constant);
    drop(rope);
    commands.end_encoding().submit().wait_until_completed().map_err(|_| "execute")?;
    let normalized: Vec<_> = copies[0].copyout::<bf16>().iter().map(|x| x.to_f32()).collect();
    let extract = |offset: u32, width: u32| -> Vec<f32> {
        normalized
            .chunks_exact(columns as usize)
            .flat_map(|row| row[offset as usize..(offset + width) as usize].iter().copied())
            .collect()
    };
    let mut stages = vec![
        (format!("layer-{layer}-q-normalized"), vec![1, rows, q_width], extract(0, q_width)),
        (format!("layer-{layer}-k-normalized"), vec![1, rows, kv_width], extract(q_width, kv_width)),
        (format!("layer-{layer}-v-projection"), vec![1, rows, kv_width], extract(q_width + kv_width, kv_width)),
        (
            format!("attention-query-{layer}"),
            vec![attention.num_heads, rows, attention.head_dim],
            copies[1].copyout::<bf16>().iter().map(|x| x.to_f32()).collect(),
        ),
        (
            if cached_step {
                format!("prepared-key-native-{layer}")
            } else {
                format!("attention-key-native-{layer}")
            },
            vec![rows, attention.num_groups, attention.head_dim],
            copies[2].copyout::<bf16>().iter().map(|x| x.to_f32()).collect(),
        ),
        (
            if cached_step {
                format!("prepared-value-native-{layer}")
            } else {
                format!("attention-value-native-{layer}")
            },
            vec![rows, attention.num_groups, attention.head_dim],
            copies[3].copyout::<bf16>().iter().map(|x| x.to_f32()).collect(),
        ),
    ];
    if direct_source_rms {
        for (index, name, width) in [(0, "q", q_width), (1, "k", kv_width)] {
            stages.push((
                format!("direct-source-rms-layer-{layer}-{name}-normalized"),
                vec![1, rows, width],
                copies[4 + index].copyout::<bf16>().iter().map(|x| x.to_f32()).collect(),
            ));
        }
    }
    if matches!(rope_config, AnyRoPEConfig::YARNRoPEConfig(_)) {
        let mscale = receipt["prepared_rope_controls"]["yarn_by_layer"][layer.to_string()]["input_mscale"]
            .as_f64()
            .or_else(|| receipt["prepared_rope_controls"]["yarn_input_mscale"].as_f64())
            .ok_or("yarn_mscale")? as f32;
        let weak_scale = bf16::from_f32(mscale).to_f32();
        let q_norm = read(&format!("layer-{layer}-q-normalized"), &[1, rows, q_width])?;
        let k_norm = read(&format!("layer-{layer}-k-normalized"), &[1, rows, kv_width])?;
        let mut q_scaled = vec![0.0; q_norm.len()];
        for head in 0..attention.num_heads as usize {
            for row in 0..rows as usize {
                for d in 0..attention.head_dim as usize {
                    let from = row * q_width as usize + head * attention.head_dim as usize + d;
                    let to = (head * rows as usize + row) * attention.head_dim as usize + d;
                    q_scaled[to] = bf16::from_f32(q_norm[from] * weak_scale).to_f32();
                }
            }
        }
        let k_scaled = k_norm.iter().map(|&x| bf16::from_f32(x * weak_scale).to_f32()).collect();
        stages.push((
            format!("layer-{layer}-q-yarn-prescaled"),
            vec![attention.num_heads, rows, attention.head_dim],
            q_scaled,
        ));
        stages.push((
            format!("layer-{layer}-k-yarn-prescaled-native"),
            vec![rows, attention.num_groups, attention.head_dim],
            k_scaled,
        ));
    }
    fs::DirBuilder::new().mode(0o700).create(&output).map_err(|_| "output_create")?;
    let mut reports = Vec::new();
    for (name, shape, native) in stages {
        let reference = read(name.strip_prefix("direct-source-rms-").unwrap_or(&name), &shape)?;
        let errors: Vec<_> = native.iter().zip(&reference).map(|(&a, &b)| f64::from(a) - f64::from(b)).collect();
        let bytes: Vec<_> = native.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(output.join(format!("{name}-native.f32")))
            .map_err(|_| "stage_create")?;
        file.write_all(&bytes).map_err(|_| "stage_write")?;
        reports.push(serde_json::json!({"component":name,"elements":native.len(),"different":native.iter().zip(&reference).filter(|(a,b)|a!=b).count(),
            "rmse":(errors.iter().map(|x|x*x).sum::<f64>()/errors.len() as f64).sqrt(),"max_abs":errors.iter().map(|x|x.abs()).fold(0.0,f64::max),
            "native_sha256":format!("{:x}",Sha256::digest(&bytes))}));
    }
    let report = serde_json::to_vec(&serde_json::json!({"backend":"metal","source_rope":source_rope,"direct_source_rms":direct_source_rms,"layer":layer,"input_tokens":rows,"position_offset":position_offset,"cached_step":cached_step,
        "fixture_receipt_sha256":format!("{:x}",Sha256::digest(&receipt_bytes)),"model_config_sha256":format!("{:x}",Sha256::digest(&config_bytes)),"components":reports})).map_err(|_| "report_encode")?;
    let mut report_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(output.join("report.json"))
        .map_err(|_| "report_create")?;
    report_file.write_all(&report).map_err(|_| "report_write")?;
    Ok(())
}

#[stark_test]
#[ignore]
fn olmo3_private_prepared_inputs_probe() {
    let result = prepared_inputs_probe();
    assert!(result.is_ok(), "private OLMo prepared-input probe failed: {}", result.err().unwrap_or("unknown"));
}
