//! Explicitly selected private Gemma operator tracing. Never run by CI.

use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::PathBuf,
};

use half::bf16;
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

#[path = "../../src/encodable_block/mixer/attention/qkv_norm.rs"]
mod production_qkv_norm;

use crate::{
    backends::{
        common::{
            Backend, Buffer, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable,
            CommandBufferPending, Context, gpu_types::trie::TrieNode,
        },
        metal::Metal,
    },
    config::decoder::DecoderConfig,
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, decoder::Decoder, linear::Linear},
    parameters::ParameterLoader,
    tests::private_output::{create_private_directory, private_write},
};

fn trace() -> Result<(), &'static str> {
    let model_path = PathBuf::from(std::env::var_os("STARK_GEMMA_TRACE_MODEL").ok_or("trace_model_missing")?);
    let fixture_path = PathBuf::from(std::env::var_os("STARK_GEMMA_TRACE_FIXTURE").ok_or("trace_fixture_missing")?);
    let output_path = PathBuf::from(std::env::var_os("STARK_GEMMA_TRACE_OUTPUT").ok_or("trace_output_missing")?);
    let raw: serde_json::Value =
        serde_json::from_slice(&fs::read(model_path.join("config.json")).map_err(|_| "config_read_failed")?)
            .map_err(|_| "config_invalid")?;
    let config: DecoderConfig =
        serde_json::from_value(raw["decoder_config"].clone()).map_err(|_| "decoder_config_invalid")?;
    let fixture: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture_path.join("cases.json")).map_err(|_| "fixture_read_failed")?)
            .map_err(|_| "fixture_invalid")?;
    let case_index = std::env::var("STARK_GEMMA_TRACE_CASE_INDEX")
        .ok()
        .map(|value| value.parse::<usize>().map_err(|_| "case_index_invalid"))
        .transpose()?
        .unwrap_or(0);
    let mut tokens: Vec<u32> =
        serde_json::from_value(fixture["cases"][case_index]["tokens"].clone()).map_err(|_| "tokens_invalid")?;
    let prefill = std::env::var_os("STARK_GEMMA_TRACE_PREFILL").is_some();
    let cached_last = std::env::var_os("STARK_GEMMA_TRACE_CACHED_LAST").is_some();
    let prefix = if cached_last {
        if tokens.len() < 2 || tokens.len() > 1024 {
            return Err("cached_trace_geometry");
        }
        let prefix = tokens[..tokens.len() - 1].to_vec();
        tokens = tokens[tokens.len() - 1..].to_vec();
        Some(prefix)
    } else {
        None
    };
    if prefill {
        tokens.pop();
    }
    if tokens.is_empty()
        || tokens.len() > 1024
        || (!prefill && !cached_last && tokens.len() != 1)
        || config.transformer_config.layer_configs.len() != 35
        || config.ple_model_config.is_none()
    {
        return Err("trace_requires_bounded_gemma4_e2b_prefix");
    }
    let batch = tokens.len() as u32;
    let context = <Metal as Backend>::Context::new().map_err(|_| "context_failed")?;
    let weights = File::open(model_path.join("model.safetensors")).map_err(|_| "weights_read_failed")?;
    let loader = ParameterLoader::<Metal>::new(&weights, &context).map_err(|_| "loader_failed")?;
    let decoder = Decoder::<Metal>::new(&context, &config, &loader.tree().subtree("decoder"), DataType::BF16)
        .map_err(|_| "decoder_failed")?;
    loader.tree().assert_all_tensors_validated().map_err(|_| "unvalidated_tensor")?;
    let mut state = decoder.create_empty_state(Some(1024), &context).map_err(|_| "state_failed")?;
    state.prepare(0, prefix.as_ref().map_or(batch, |p| p.len() as u32 + 1)).map_err(|_| "state_prepare_failed")?;
    let mut commands =
        context.create_command_buffer(Some("private Gemma precision trace"), None).map_err(|_| "commands_failed")?;
    if let Some(prefix) = &prefix {
        let prefix_tokens = commands.allocate_constant_from_slice(prefix).map_err(|_| "prefix_tokens")?;
        let nodes: Vec<_> = (0..prefix.len() as u32)
            .map(|index| TrieNode {
                trie_start: index,
                trie_end: prefix.len() as u32 - 1,
                height: index,
            })
            .collect();
        let topology = BatchTopology::new(&nodes, true);
        decoder
            .encode(&prefix_tokens, &topology, None, None, &mut state, &mut commands)
            .map_err(|_| "prefix_encode")?;
        state
            .encode_accept(&(0..prefix.len() as u32).collect::<Box<[_]>>(), &mut commands)
            .map_err(|_| "prefix_accept")?;
    }
    let token_ids = commands.allocate_constant_from_slice(&tokens).map_err(|_| "token_buffer_failed")?;
    let embedding =
        decoder.embedding().encode_lookup(&token_ids, batch, &mut commands).map_err(|_| "embedding_failed")?;
    let nodes: Vec<_> = (0..batch)
        .map(|index| TrieNode {
            trie_start: index,
            trie_end: batch - 1,
            height: index,
        })
        .collect();
    let topology = BatchTopology::new(&nodes, true);
    let layers: Vec<u32> = (0..35).collect();
    let output = decoder
        .encode(&token_ids, &topology, Some((batch - 1..batch).into()), Some(&layers), &mut state, &mut commands)
        .map_err(|_| "decode_failed")?;
    let mut sources = vec![("embedding".to_owned(), embedding)];
    for (index, hidden) in output.hidden_features.ok_or("hidden_missing")?.into_vec().into_iter().enumerate() {
        sources.push((format!("layer-{index}"), hidden));
    }
    sources.push(("final-hidden".to_owned(), output.final_hidden.ok_or("final_hidden_missing")?));
    sources.push(("logits".to_owned(), output.logits.ok_or("logits_missing")?));
    let mut readbacks = Vec::new();
    for (name, source) in sources {
        let mut destination = context.create_buffer(source.size()).map_err(|_| "readback_failed")?;
        commands.encode_copy(&source, &mut destination);
        readbacks.push((name, destination));
    }
    drop(token_ids);
    commands.end_encoding().submit().wait_until_completed().map_err(|_| "execution_failed")?;
    create_private_directory(&output_path).map_err(|_| "output_create_failed")?;
    let mut rows = Vec::new();
    for (name, buffer) in readbacks {
        let values: Vec<bf16> = buffer.copyout();
        if values.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite_trace");
        }
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_f32().to_le_bytes()).collect();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(output_path.join(format!("{name}.f32")))
            .map_err(|_| "trace_file_create_failed")?;
        file.write_all(&bytes).map_err(|_| "trace_write_failed")?;
        rows.push(serde_json::json!({"component":name,"elements":values.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}));
    }
    private_write(
        &output_path.join("report.json"),
        &serde_json::to_vec(&serde_json::json!({"backend":"metal","precision":"bf16","input_tokens":batch,"case_index":case_index,"prefill_prefix":prefill,"cached_last":cached_last,"prefix_tokens":prefix.as_ref().map_or(0,Vec::len),"components":rows}))
            .map_err(|_| "report_encode_failed")?,
    )
    .map_err(|_| "report_write_failed")?;
    Ok(())
}

#[stark_test]
#[ignore]
fn gemma4_private_precision_trace() {
    let result = trace();
    assert!(result.is_ok(), "private Gemma precision trace failed: {}", result.err().unwrap_or("unknown"));
}

fn prefill_projection_probe() -> Result<(), &'static str> {
    let model = PathBuf::from(std::env::var_os("STARK_GEMMA_TRACE_MODEL").ok_or("model_missing")?);
    let fixture = PathBuf::from(std::env::var_os("STARK_GEMMA_PROJECTION_FIXTURE").ok_or("fixture_missing")?);
    let output = PathBuf::from(std::env::var_os("STARK_GEMMA_PROJECTION_OUTPUT").ok_or("output_missing")?);
    let raw: serde_json::Value =
        serde_json::from_slice(&fs::read(model.join("config.json")).map_err(|_| "config_read_failed")?)
            .map_err(|_| "config_invalid")?;
    let config: DecoderConfig = serde_json::from_value(raw["decoder_config"].clone()).map_err(|_| "config_invalid")?;
    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture.join("receipt.json")).map_err(|_| "receipt_read_failed")?)
            .map_err(|_| "receipt_invalid")?;
    if receipt["device"] != "gpu"
        || receipt["identity"]["mlx_lm_revision"] != "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"
    {
        return Err("source_identity_mismatch");
    }
    let read = |name: &str| -> Result<Vec<f32>, &'static str> {
        let bytes = fs::read(fixture.join(format!("{name}.f32"))).map_err(|_| "tensor_read_failed")?;
        let item = receipt["operations"]
            .as_array()
            .ok_or("operations_missing")?
            .iter()
            .find(|item| item["operation"] == name)
            .ok_or("source_tensor_missing")?;
        if bytes.len() % 4 != 0 || item["sha256"] != format!("{:x}", Sha256::digest(&bytes)) {
            return Err("source_tensor_hash_mismatch");
        }
        Ok(bytes.as_chunks::<4>().0.iter().map(|x| f32::from_le_bytes(*x)).collect())
    };
    let input = read("actual_qkv_input")?;
    let expected = read("actual_qkv")?;
    let width = config.transformer_config.model_dim;
    if input.len() % width as usize != 0 {
        return Err("input_shape_mismatch");
    }
    let rows = input.len() as u32 / width;
    if !(1..=1024).contains(&rows) || expected.len() % rows as usize != 0 {
        return Err("projection_shape_mismatch");
    }
    let columns = expected.len() as u32 / rows;
    let mixer = &raw["decoder_config"]["transformer_config"]["layer_configs"][0]["mixer_config"];
    let heads = mixer["num_heads"].as_u64().ok_or("heads_missing")? as u32;
    let groups = mixer["num_groups"].as_u64().ok_or("groups_missing")? as u32;
    let head_dim = mixer["head_dim"].as_u64().ok_or("head_dimension_missing")? as u32;
    let dimensions = [heads * head_dim, groups * head_dim, groups * head_dim];
    if dimensions.iter().sum::<u32>() != columns {
        return Err("projection_dimensions_mismatch");
    }
    let context = <Metal as Backend>::Context::new().map_err(|_| "context_failed")?;
    let weights = File::open(model.join("model.safetensors")).map_err(|_| "weights_read_failed")?;
    let loader = ParameterLoader::<Metal>::new(&weights, &context).map_err(|_| "loader_failed")?;
    let projection = <dyn Linear<Metal>>::new(
        width,
        dimensions,
        false,
        &context,
        DataType::BF16,
        &loader.tree().subtree("decoder.transformer.layers.0.mixer.qkvg_projection"),
    )
    .map_err(|_| "projection_load_failed")?;
    let mut commands = context
        .create_command_buffer(Some("private Gemma prefill projection control"), None)
        .map_err(|_| "commands_failed")?;
    let values: Vec<bf16> = input.iter().map(|x| bf16::from_f32(*x)).collect();
    let constant = commands.allocate_constant_from_slice(&values).map_err(|_| "input_failed")?;
    let mut scratch =
        commands.allocate_scratch_for_shape(&[rows, width], DataType::BF16).map_err(|_| "scratch_failed")?;
    commands.encode_copy(&constant, &mut scratch);
    let actual = projection.encode(scratch, rows, &mut commands).map_err(|_| "projection_failed")?;
    let mut readback = context.create_buffer(actual.size()).map_err(|_| "readback_failed")?;
    commands.encode_copy(&actual, &mut readback);
    drop(constant);
    commands.end_encoding().submit().wait_until_completed().map_err(|_| "execution_failed")?;
    let actual: Vec<bf16> = readback.copyout();
    if actual.len() != expected.len() {
        return Err("output_shape_mismatch");
    }
    let differences: Vec<f64> = actual.iter().zip(&expected).map(|(a, b)| a.to_f64() - f64::from(*b)).collect();
    let different = differences.iter().filter(|x| **x != 0.0).count();
    let maximum = differences.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let rmse = (differences.iter().map(|x| x * x).sum::<f64>() / differences.len() as f64).sqrt();
    let bytes: Vec<u8> = actual.iter().flat_map(|x| x.to_f32().to_le_bytes()).collect();
    create_private_directory(&output).map_err(|_| "output_create_failed")?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(output.join("native-qkv.f32"))
        .map_err(|_| "output_file_failed")?;
    file.write_all(&bytes).map_err(|_| "output_write_failed")?;
    private_write(&output.join("report.json"),& serde_json::to_vec(&serde_json::json!({"backend":"metal","rows":rows,"input_width":width,"output_width":columns,"elements":actual.len(),"different":different,"max_abs":maximum,"rmse":rmse,"native_sha256":format!("{:x}",Sha256::digest(&bytes))})).map_err(|_| "report_invalid")?).map_err(|_| "report_write_failed")?;
    Ok(())
}

#[stark_test]
#[ignore]
fn gemma4_private_prefill_projection_probe() {
    let result = prefill_projection_probe();
    assert!(result.is_ok(), "private Gemma projection probe failed: {}", result.err().unwrap_or("unknown"));
}

fn prepared_inputs_probe() -> Result<(), &'static str> {
    use crate::{
        backends::common::kernel::{AttentionPrepareKernel, Kernels},
        config::token_mixer::attention::AttentionConfig,
        encodable_block::mixer::attention::rope::PrecalculatedRoPE,
    };
    let model = PathBuf::from(std::env::var_os("STARK_GEMMA_TRACE_MODEL").ok_or("model_missing")?);
    let fixture = PathBuf::from(std::env::var_os("STARK_GEMMA_PROJECTION_FIXTURE").ok_or("fixture_missing")?);
    let output = PathBuf::from(std::env::var_os("STARK_GEMMA_PROJECTION_OUTPUT").ok_or("output_missing")?);
    let raw: serde_json::Value =
        serde_json::from_slice(&fs::read(model.join("config.json")).map_err(|_| "config_read")?)
            .map_err(|_| "config_invalid")?;
    let config: DecoderConfig = serde_json::from_value(raw["decoder_config"].clone()).map_err(|_| "decoder_invalid")?;
    let receipt_bytes = fs::read(fixture.join("receipt.json")).map_err(|_| "receipt_read")?;
    let receipt: serde_json::Value = serde_json::from_slice(&receipt_bytes).map_err(|_| "receipt_invalid")?;
    if receipt["device"] != "gpu"
        || receipt["identity"]["mlx_lm_revision"] != "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"
    {
        return Err("source_identity_mismatch");
    }
    let layer = receipt["identity"]["actual_layer_index"].as_u64().unwrap_or(0) as usize;
    if layer >= config.transformer_config.layer_configs.len() {
        return Err("layer_out_of_bounds");
    }
    let mut attention: AttentionConfig = serde_json::from_value(
        raw["decoder_config"]["transformer_config"]["layer_configs"][layer]["mixer_config"].clone(),
    )
    .map_err(|_| "attention_invalid")?;
    let read = |name: &str| -> Result<Vec<f32>, &'static str> {
        let bytes = fs::read(fixture.join(format!("{name}.f32"))).map_err(|_| "tensor_read")?;
        let entry = receipt["operations"]
            .as_array()
            .ok_or("operations_missing")?
            .iter()
            .find(|entry| entry["operation"] == name)
            .ok_or("operation_missing")?;
        if !bytes.len().is_multiple_of(4) || entry["sha256"] != format!("{:x}", Sha256::digest(&bytes)) {
            return Err("tensor_hash");
        }
        let values: Vec<_> = bytes.as_chunks::<4>().0.iter().map(|x| f32::from_le_bytes(*x)).collect();
        if values.iter().any(|&x| !x.is_finite() || bf16::from_f32(x).to_f32() != x) {
            return Err("source_not_bf16");
        }
        Ok(values)
    };
    if std::env::var_os("STARK_GEMMA_SOURCE_QKV_RMS").is_some() {
        if let Some(config) = &mut attention.query_norm_config {
            config.mlx_bf16_reduction = true;
        }
        if let Some(config) = &mut attention.key_norm_config {
            config.mlx_bf16_reduction = true;
        }
    }
    let input = read("actual_qkv")?;
    let q_width = attention.num_heads * attention.head_dim;
    let kv_width = attention.num_groups * attention.head_dim;
    let has_kv = !attention.is_kv_sharing;
    let columns = q_width
        + if has_kv {
            2 * kv_width
        } else {
            0
        };
    if input.len() % columns as usize != 0 {
        return Err("source_shape");
    }
    let rows = input.len() as u32 / columns;
    if !(1..=1024).contains(&rows) {
        return Err("source_rows");
    }
    let context = <Metal as Backend>::Context::new().map_err(|_| "context")?;
    let weights = File::open(model.join("model.safetensors")).map_err(|_| "weights_read")?;
    let loader = ParameterLoader::<Metal>::new(&weights, &context).map_err(|_| "loader")?;
    let norm = production_qkv_norm::QKVNorm::<Metal>::new(
        &context,
        DataType::BF16,
        attention.query_norm_config.clone(),
        if has_kv {
            attention.key_norm_config.clone()
        } else {
            None
        },
        if has_kv {
            attention.value_norm_config()
        } else {
            None
        },
        &loader.tree().subtree(&format!("decoder.transformer.layers.{layer}.mixer")),
        attention.num_heads,
        if has_kv {
            attention.num_groups
        } else {
            0
        },
        columns,
        attention.head_dim,
        attention.qk_norm_full_width,
    )
    .map_err(|_| "norm_new")?;
    let mut commands = context.create_command_buffer(None, None).map_err(|_| "commands")?;
    let input_bf16: Vec<_> = input.iter().map(|&x| bf16::from_f32(x)).collect();
    let constant = commands.allocate_constant_from_slice(&input_bf16).map_err(|_| "constant")?;
    let mut normalized =
        commands.allocate_scratch_for_shape(&[rows, columns], DataType::BF16).map_err(|_| "scratch")?;
    commands.encode_copy(&constant, &mut normalized);
    norm.encode(&mut normalized, rows, &mut commands).map_err(|_| "norm_encode")?;
    let position_offset = receipt["identity"]["actual_position_offset"].as_u64().unwrap_or(0) as u32;
    let positions: Vec<_> = (position_offset..position_offset + rows).collect();
    let rope = PrecalculatedRoPE::<Metal>::precalculate(
        config.transformer_config.layer_configs[layer].rope_config.as_ref().ok_or("rope_missing")?,
        &positions,
        &mut commands,
    )
    .map_err(|_| "rope_tables")?;
    let source_rope = std::env::var("STARK_GEMMA_SOURCE_ROPE").ok().as_deref() == Some("1");
    let source_parameters = rope.source_parameters.ok_or("source_parameters")?;
    let prepare = <<Metal as Backend>::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        DataType::BF16,
        DataType::F32,
        has_kv,
        true,
        attention.reference_bf16,
        source_rope,
        source_rope && source_parameters.proportional,
        source_rope && source_parameters.yarn,
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
        has_kv.then_some(&mut keys),
        has_kv.then_some(&mut values),
        Some(&rope.cosines),
        Some(&rope.sines),
        if source_rope {
            rope.source_positions.as_ref()
        } else {
            None
        },
        source_rope.then_some(source_parameters.base),
        source_rope.then_some(source_parameters.rotary_dim),
        if source_rope {
            rope.source_yarn_parameters.as_ref()
        } else {
            None
        },
        attention.num_heads,
        has_kv.then_some(attention.num_groups),
        attention.head_dim,
        Some(rope.dim),
        has_kv.then_some(0),
        columns,
        rows,
        &mut commands,
    );
    let mut copies = Vec::new();
    let mut sources = vec![&normalized, &queries];
    if has_kv {
        sources.extend([&keys, &values]);
    }
    for source in sources {
        let mut copy = context.create_buffer(source.size()).map_err(|_| "readback")?;
        commands.encode_copy(source, &mut copy);
        copies.push(copy);
    }
    drop(constant);
    drop(rope);
    commands.end_encoding().submit().wait_until_completed().map_err(|_| "execute")?;
    let normalized: Vec<f32> = copies[0].copyout::<bf16>().iter().map(|x| x.to_f32()).collect();
    let extract = |start: u32, width: u32| -> Vec<f32> {
        normalized
            .chunks_exact(columns as usize)
            .flat_map(|row| row[start as usize..(start + width) as usize].iter().copied())
            .collect()
    };
    let mut stages = vec![
        ("actual_q_norm", extract(0, q_width)),
        ("actual_q_rope", copies[1].copyout::<bf16>().iter().map(|x| x.to_f32()).collect()),
    ];
    if has_kv {
        stages.extend([
            ("actual_k_norm", extract(q_width, kv_width)),
            ("actual_v_norm", extract(q_width + kv_width, kv_width)),
            ("actual_k_rope_native", copies[2].copyout::<bf16>().iter().map(|x| x.to_f32()).collect()),
        ]);
    }
    create_private_directory(&output).map_err(|_| "output_create")?;
    let mut reports = Vec::new();
    for (name, native) in stages {
        let reference = read(name)?;
        if native.len() != reference.len() {
            return Err("stage_shape");
        }
        let error: Vec<_> = native.iter().zip(&reference).map(|(&a, &b)| f64::from(a) - f64::from(b)).collect();
        let bytes: Vec<_> = native.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(output.join(format!("{name}-native.f32")))
            .map_err(|_| "stage_create")?;
        file.write_all(&bytes).map_err(|_| "stage_write")?;
        reports.push(serde_json::json!({"component":name,"elements":native.len(),"different":native.iter().zip(&reference).filter(|(a,b)|a!=b).count(),
            "rmse":(error.iter().map(|x|x*x).sum::<f64>()/error.len() as f64).sqrt(),"max_abs":error.iter().map(|x|x.abs()).fold(0.0,f64::max),
            "native_sha256":format!("{:x}",Sha256::digest(&bytes))}));
    }
    private_write(&output.join("report.json"),&serde_json::to_vec(&serde_json::json!({"backend":"metal","layer":layer,"source_rope":source_rope,"input_tokens":rows,"fixture_receipt_sha256":format!("{:x}",Sha256::digest(&receipt_bytes)),"components":reports})).map_err(|_|"report_encode")?).map_err(|_|"report_write")?;
    Ok(())
}

#[stark_test]
#[ignore]
fn gemma4_private_prepared_inputs_probe() {
    let result = prepared_inputs_probe();
    assert!(result.is_ok(), "private Gemma prepared-input probe failed: {}", result.err().unwrap_or("unknown"));
}

fn operator_probe() -> Result<(), &'static str> {
    use crate::{
        config::normalization::NormalizationConfig,
        encodable_block::normalization::{Normalization, PostLayerScalar, ShortcutMode},
    };
    let model = PathBuf::from(std::env::var_os("STARK_GEMMA_TRACE_MODEL").ok_or("model_missing")?);
    let fixture = PathBuf::from(std::env::var_os("STARK_GEMMA_PROJECTION_FIXTURE").ok_or("fixture_missing")?);
    let output = PathBuf::from(std::env::var_os("STARK_GEMMA_PROJECTION_OUTPUT").ok_or("output_missing")?);
    let raw: serde_json::Value =
        serde_json::from_slice(&fs::read(model.join("config.json")).map_err(|_| "config_read")?)
            .map_err(|_| "config_parse")?;
    let receipt_bytes = fs::read(fixture.join("receipt.json")).map_err(|_| "receipt_read")?;
    let receipt: serde_json::Value = serde_json::from_slice(&receipt_bytes).map_err(|_| "receipt_parse")?;
    if receipt["device"] != "gpu"
        || receipt["identity"]["mlx_lm_revision"] != "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"
    {
        return Err("source_identity");
    }
    let layer = receipt["identity"]["actual_layer_index"].as_u64().ok_or("layer_missing")? as usize;
    let rows = receipt["identity"]["actual_projection_rows"].as_u64().ok_or("rows_missing")? as u32;
    if layer >= 35 || !(1..=1024).contains(&rows) {
        return Err("geometry");
    }
    let layer_config = &raw["decoder_config"]["transformer_config"]["layer_configs"][layer];
    let read = |name: &str| -> Result<Vec<f32>, &'static str> {
        let bytes = fs::read(fixture.join(format!("{name}.f32"))).map_err(|_| "tensor_read")?;
        let item = receipt["operations"]
            .as_array()
            .ok_or("operations_missing")?
            .iter()
            .find(|item| item["operation"] == name)
            .ok_or("tensor_missing")?;
        if !bytes.len().is_multiple_of(4) || item["sha256"] != format!("{:x}", Sha256::digest(&bytes)) {
            return Err("tensor_hash");
        }
        let values: Vec<_> = bytes.as_chunks::<4>().0.iter().map(|x| f32::from_le_bytes(*x)).collect();
        if values.iter().any(|&x| !x.is_finite() || bf16::from_f32(x).to_f32() != x) {
            return Err("tensor_dtype");
        }
        Ok(values)
    };
    let context = <Metal as Backend>::Context::new().map_err(|_| "context")?;
    let weights = File::open(model.join("model.safetensors")).map_err(|_| "weights")?;
    let loader = ParameterLoader::<Metal>::new(&weights, &context).map_err(|_| "loader")?;
    let tree = loader.tree().subtree(&format!("decoder.transformer.layers.{layer}"));
    create_private_directory(&output).map_err(|_| "output_create")?;
    let descriptors: &[(&str, &str, &[&str], &str, Option<&str>)] = &[
        (
            "input_norm",
            "actual_input_norm_input",
            &["actual_input_norm"],
            "pre_mixer_norm",
            Some("pre_mixer_norm_config"),
        ),
        (
            "qkv",
            "actual_qkv_input",
            &["actual_q_projection", "actual_k_projection", "actual_v_projection"],
            "mixer.qkvg_projection",
            None,
        ),
        ("o_projection", "actual_o_projection_input", &["actual_o_projection"], "mixer.out_projection", None),
        (
            "post_attention_norm",
            "actual_post_attention_norm_input",
            &["actual_post_attention_norm"],
            "post_mixer_norm",
            Some("post_mixer_norm_config"),
        ),
        ("mlp_norm", "actual_mlp_norm_input", &["actual_mlp_norm"], "pre_mlp_norm", Some("pre_mlp_norm_config")),
        ("mlp_up_gate", "actual_mlp_up_input", &["actual_mlp_up", "actual_mlp_gate"], "mlp.up_projection", None),
        ("mlp_down", "actual_mlp_down_input", &["actual_mlp_down"], "mlp.down_projection", None),
        (
            "post_mlp_norm",
            "actual_post_mlp_norm_input",
            &["actual_post_mlp_norm"],
            "post_mlp_norm",
            Some("post_mlp_norm_config"),
        ),
        ("ple_gate", "actual_ple_gate_input", &["actual_ple_gate"], "ple.gate", None),
        ("ple_projection", "actual_ple_projection_input", &["actual_ple_projection"], "ple.projection", None),
        ("ple_norm", "actual_ple_norm_input", &["actual_ple_norm"], "ple.norm", Some("ple_config")),
    ];
    let mut reports = Vec::new();
    for &(name, input_name, expected_names, path, norm_key) in descriptors {
        let expected_names = if name == "qkv" && layer_config["mixer_config"]["is_kv_sharing"] == true {
            &["actual_q_projection"][..]
        } else {
            expected_names
        };
        let input = read(input_name)?;
        if !input.len().is_multiple_of(rows as usize) {
            return Err("input_shape");
        }
        let input_width = input.len() as u32 / rows;
        let references = expected_names.iter().map(|name| read(name)).collect::<Result<Vec<_>, _>>()?;
        if references.iter().any(|a| !a.len().is_multiple_of(rows as usize)) {
            return Err("output_shape");
        }
        let widths: Vec<_> = references.iter().map(|a| a.len() as u32 / rows).collect();
        let mut expected = Vec::new();
        for row in 0..rows as usize {
            for (a, &width) in references.iter().zip(&widths) {
                expected.extend_from_slice(&a[row * width as usize..(row + 1) * width as usize]);
            }
        }
        let mut commands = context.create_command_buffer(None, None).map_err(|_| "commands")?;
        let input_bf16: Vec<_> = input.iter().copied().map(bf16::from_f32).collect();
        let constant = commands.allocate_constant_from_slice(&input_bf16).map_err(|_| "constant")?;
        let mut scratch =
            commands.allocate_scratch_for_shape(&[rows, input_width], DataType::BF16).map_err(|_| "scratch")?;
        commands.encode_copy(&constant, &mut scratch);
        let actual = if let Some(key) = norm_key {
            let mut norm_config: NormalizationConfig = serde_json::from_value(if key == "ple_config" {
                layer_config[key]["norm_config"].clone()
            } else {
                layer_config[key].clone()
            })
            .map_err(|_| "norm_config")?;
            if std::env::var_os("STARK_GEMMA_SOURCE_RMS").is_some() {
                norm_config.mlx_bf16_reduction = true;
            }
            let norm = Normalization::<Metal>::new(
                input_width,
                None,
                ShortcutMode::None,
                PostLayerScalar::None,
                DataType::BF16,
                &norm_config,
                &tree.subtree(path),
                &context,
            )
            .map_err(|_| "norm_new")?;
            norm.encode(&scratch, 0, rows, None::<&mut <Metal as Backend>::ScratchBuffer>, &mut commands)
                .map_err(|_| "norm_encode")?
        } else {
            let linear =
                <dyn Linear<Metal>>::new(input_width, &widths, false, &context, DataType::BF16, &tree.subtree(path))
                    .map_err(|_| "linear_new")?;
            linear.encode(scratch, rows, &mut commands).map_err(|_| "linear_encode")?
        };
        let mut copy = context.create_buffer(actual.size()).map_err(|_| "copy")?;
        commands.encode_copy(&actual, &mut copy);
        drop(constant);
        commands.end_encoding().submit().wait_until_completed().map_err(|_| "execute")?;
        let native: Vec<_> = copy.copyout::<bf16>().into_iter().map(|x| x.to_f32()).collect();
        if native.len() != expected.len() {
            return Err("native_shape");
        }
        let error: Vec<_> = native.iter().zip(&expected).map(|(&a, &b)| f64::from(a) - f64::from(b)).collect();
        let bytes: Vec<_> = native.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(output.join(format!("{name}-native.f32")))
            .map_err(|_| "output_file")?;
        file.write_all(&bytes).map_err(|_| "output_write")?;
        reports.push(serde_json::json!({"component":name,"elements":native.len(),"different":error.iter().filter(|x|**x!=0.0).count(),"rmse":(error.iter().map(|x|x*x).sum::<f64>()/error.len() as f64).sqrt(),"max_abs":error.iter().map(|x|x.abs()).fold(0.0,f64::max),"native_sha256":format!("{:x}",Sha256::digest(&bytes))}));
    }
    private_write(&output.join("report.json"),&serde_json::to_vec(&serde_json::json!({"backend":"metal","layer":layer,"rows":rows,"fixture_receipt_sha256":format!("{:x}",Sha256::digest(&receipt_bytes)),"components":reports})).map_err(|_|"report_encode")?).map_err(|_|"report_write")?;
    Ok(())
}

#[stark_test]
#[ignore]
fn gemma4_private_operator_probe() {
    let result = operator_probe();
    assert!(result.is_ok(), "private Gemma operator probe failed: {}", result.err().unwrap_or("unknown"));
}

fn attention_stages_probe() -> Result<(), &'static str> {
    use crate::{
        backends::common::{
            Kernels,
            kernel::{AttentionArguments, AttentionKernelConfig},
        },
        encodable_block::mixer::attention::KVCacheView,
    };
    let fixture = PathBuf::from(std::env::var_os("STARK_GEMMA_PROJECTION_FIXTURE").ok_or("fixture_missing")?);
    let output = PathBuf::from(std::env::var_os("STARK_GEMMA_PROJECTION_OUTPUT").ok_or("output_missing")?);
    let receipt_bytes = fs::read(fixture.join("receipt.json")).map_err(|_| "receipt_read")?;
    let receipt: serde_json::Value = serde_json::from_slice(&receipt_bytes).map_err(|_| "receipt_parse")?;
    if receipt["device"] != "gpu"
        || receipt["identity"]["mlx_lm_revision"] != "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"
    {
        return Err("source_identity");
    }
    let rows = receipt["identity"]["actual_projection_rows"].as_u64().ok_or("rows_missing")? as usize;
    if !(1..1024).contains(&rows) {
        return Err("source_rows");
    }
    let read = |name: &str| -> Result<Vec<f32>, &'static str> {
        let data = fs::read(fixture.join(format!("{name}.f32"))).map_err(|_| "tensor_read")?;
        let entry = receipt["operations"]
            .as_array()
            .ok_or("operations_missing")?
            .iter()
            .find(|e| e["operation"] == name)
            .ok_or("tensor_missing")?;
        if !data.len().is_multiple_of(4) || entry["sha256"] != format!("{:x}", Sha256::digest(&data)) {
            return Err("tensor_hash");
        }
        let values: Vec<_> = data.as_chunks::<4>().0.iter().map(|x| f32::from_le_bytes(*x)).collect();
        if values.iter().any(|&x| !x.is_finite() || bf16::from_f32(x).to_f32() != x) {
            return Err("tensor_dtype");
        }
        Ok(values)
    };
    let context = <Metal as Backend>::Context::new().map_err(|_| "context")?;
    let q = read("actual_q_rope")?;
    let k = read("actual_attention_keys_native")?;
    let v = read("actual_attention_values_native")?;
    let head_dim = q.len() / (8 * rows);
    if !matches!(head_dim, 256 | 512) {
        return Err("head_dimension");
    }
    let sequence = k.len() / head_dim;
    if !matches!(head_dim, 256 | 512)
        || q.len() != 8 * rows * head_dim
        || k.len() != sequence * head_dim
        || sequence < rows
        || (rows == 1 && head_dim != 512)
        || v.len() != k.len()
    {
        return Err("source_shape");
    }
    let to_buffer = |values: &[f32]| {
        crate::tests::helpers::create_buffer_with_data::<Metal, bf16>(
            &context,
            &values.iter().copied().map(bf16::from_f32).collect::<Vec<_>>(),
        )
    };
    let q = to_buffer(&q);
    let k = to_buffer(&k);
    let v = to_buffer(&v);
    let config = AttentionKernelConfig {
        head_dim: head_dim as u32,
        num_groups: 1,
        num_q_heads: 8,
        has_sinks: false,
        is_kv_cache_ring: head_dim == 256,
        is_causal: true,
        sliding_window_size: (head_dim == 256).then_some(512),
        scale: Some(1.0),
        data_type: DataType::BF16,
        materialize_prefill_scores: true,
        source_bf16_prefill: false,
    };
    let mut commands = context.create_command_buffer(None, None).map_err(|_| "commands")?;
    let (native, mut stages) =
        <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::encode_fallback_trace_for_test(
            &context,
            config,
            AttentionArguments {
                queries: &q,
                keys: &k,
                values: &v,
                suffix_length: rows as u32,
                trie: None::<&<Metal as Backend>::ConstantBuffer>,
                sinks: None,
                cache: if head_dim == 256 {
                    KVCacheView::ring(0, 0)
                } else {
                    KVCacheView::full((sequence - rows) as u32)
                },
            },
            &mut commands,
        )
        .map_err(|_| "attention_trace")?;
    // Independent identical-probability PV control: bypass only softmax.
    // This distinguishes a probability rounding error from full-precision GEMM.
    let source_probabilities_input = to_buffer(&read("actual_attention_probabilities")?);
    {
        use crate::backends::common::kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel};
        let mut product = commands
            .allocate_scratch_for_shape(&[8 * rows as u32, head_dim as u32], DataType::BF16)
            .map_err(|_| "source_pv_scratch")?;
        let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
            &context,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
        )
        .map_err(|_| "source_pv_kernel")?;
        kernel
            .encode(
                MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: &source_probabilities_input,
                        offset: 0,
                    },
                    b: MatmulB::FullPrecision {
                        b: &v,
                    },
                    b_leading_dimension: Some(head_dim as u32),
                    b_transpose: false,
                    d: &mut product,
                    d_transform: MatmulDOps::none(),
                    gather_indices: None::<&<Metal as Backend>::ScratchBuffer>,
                    m: 8 * rows as u32,
                    n: head_dim as u32,
                    k: sequence as u32,
                },
                &mut commands,
            )
            .map_err(|_| "source_pv_encode")?;
        stages.push(("pv_source_probabilities", product));
    }
    // Compare geometry, engines, and source F32 split accumulation on the same
    // BF16 Q/K inputs before any mask. Contiguous source head batches can collapse.
    {
        use crate::backends::{
            common::kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            metal::GemmEngine,
        };
        for (name, per_head, engine) in [
            ("qk_flat_auto", false, None),
            ("qk_flat_simd", false, Some(GemmEngine::Simdgroup)),
            ("qk_flat_mxu", false, Some(GemmEngine::Mxu)),
            ("qk_heads_auto", true, None),
            ("qk_heads_simd", true, Some(GemmEngine::Simdgroup)),
            ("qk_heads_mxu", true, Some(GemmEngine::Mxu)),
            ("qk_split2_f32", false, Some(GemmEngine::Mxu)),
        ] {
            let output_dtype = if name == "qk_split2_f32" {
                DataType::F32
            } else {
                DataType::BF16
            };
            let mut product = commands
                .allocate_scratch_for_shape(&[8 * rows as u32, sequence as u32], output_dtype)
                .map_err(|_| "qk_scratch")?;
            let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
                &context,
                DataType::BF16,
                DataType::BF16,
                output_dtype,
            )
            .map_err(|_| "qk_kernel")?;
            let batches = if per_head {
                8
            } else {
                1
            };
            let batch_rows = if per_head {
                rows
            } else {
                8 * rows
            };
            for batch in 0..batches {
                let output_begin = batch * batch_rows * sequence * output_dtype.size_in_bytes();
                let arguments = MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: &q,
                        offset: batch * batch_rows * head_dim * 2,
                    },
                    b: MatmulB::FullPrecision {
                        b: &k,
                    },
                    b_leading_dimension: Some(head_dim as u32),
                    b_transpose: true,
                    d: (&mut product).subrange_mut(
                        output_begin..output_begin + batch_rows * sequence * output_dtype.size_in_bytes(),
                    ),
                    d_transform: MatmulDOps::none(),
                    gather_indices: None::<&<Metal as Backend>::ScratchBuffer>,
                    m: batch_rows as u32,
                    n: sequence as u32,
                    k: head_dim as u32,
                };
                if name == "qk_split2_f32" {
                    kernel
                        .encode_with_gemm_split_for_test(arguments, GemmEngine::Mxu, 2, &mut commands)
                        .map_err(|_| "qk_split")?;
                } else if let Some(engine) = engine {
                    kernel.encode_with_gemm_engine(arguments, engine, &mut commands).map_err(|_| "qk_forced")?;
                } else {
                    kernel.encode(arguments, &mut commands).map_err(|_| "qk_auto")?;
                }
            }
            stages.push((name, product));
        }
    }
    stages.push(("output", native));
    let mut copies = Vec::new();
    for (name, source) in stages {
        let mut buffer = context.create_buffer(source.size()).map_err(|_| "readback")?;
        commands.encode_copy(&source, &mut buffer);
        copies.push((name, buffer));
    }
    commands.end_encoding().submit().wait_until_completed().map_err(|_| "execute")?;
    create_private_directory(&output).map_err(|_| "output_create")?;
    let mut reports = Vec::new();
    for (name, buffer) in copies {
        let source_name = match name {
            "scores" | "qk_flat_auto" | "qk_flat_simd" | "qk_flat_mxu" | "qk_heads_auto" | "qk_heads_simd"
            | "qk_split2_f32" | "qk_heads_mxu" => "actual_attention_scores",
            "probabilities" => "actual_attention_probabilities",
            "pv" | "pv_source_probabilities" => "actual_attention_reconstructed",
            _ => "actual_attention_output",
        };
        let reference = read(source_name)?;
        let native: Vec<f32> = if name == "qk_split2_f32" {
            buffer.copyout::<f32>().into_iter().map(|x| bf16::from_f32(x).to_f32()).collect()
        } else {
            buffer.copyout::<bf16>().into_iter().map(|x| x.to_f32()).collect()
        };
        if native.len() != reference.len() {
            return Err("stage_shape");
        }
        let mut error = Vec::new();
        let mut excluded = 0;
        for (i, (&a, &b)) in native.iter().zip(&reference).enumerate() {
            if (name == "scores" || name.starts_with("qk_")) && i % sequence > sequence - rows + (i / sequence) % rows {
                excluded += 1;
                continue;
            }
            if !a.is_finite() {
                return Err("stage_nonfinite");
            }
            error.push(f64::from(a) - f64::from(b));
        }
        let bytes: Vec<u8> = native.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(output.join(format!("{name}-native.f32")))
            .map_err(|_| "output_file")?;
        file.write_all(&bytes).map_err(|_| "output_write")?;
        reports.push(serde_json::json!({"component":name,"elements":native.len(),"compared_elements":error.len(),
            "excluded_equivalent_future_mask":excluded,"different":error.iter().filter(|x|**x!=0.0).count(),
            "rmse":(error.iter().map(|x|x*x).sum::<f64>()/error.len() as f64).sqrt(),"max_abs":error.iter().map(|x|x.abs()).fold(0.0,f64::max),
            "native_sha256":format!("{:x}",Sha256::digest(&bytes))}));
    }
    private_write(&output.join("report.json"),&serde_json::to_vec(&serde_json::json!({"backend":"metal",
        "source_layer_index":receipt["identity"]["actual_layer_index"],"fixture_receipt_sha256":format!("{:x}",Sha256::digest(&receipt_bytes)),
        "components":reports})).map_err(|_|"report_encode")?).map_err(|_|"report_write")?;
    Ok(())
}

#[stark_test]
#[ignore]
fn gemma4_private_attention_stages_probe() {
    let result = attention_stages_probe();
    assert!(result.is_ok(), "private Gemma attention stages probe failed: {}", result.err().unwrap_or("unknown"));
}

fn cached_vector_attention_probe() -> Result<(), &'static str> {
    use crate::{
        backends::common::{
            Kernels,
            kernel::{AttentionArguments, AttentionKernel, AttentionKernelConfig},
        },
        encodable_block::mixer::attention::KVCacheView,
    };
    let fixture = PathBuf::from(std::env::var_os("STARK_GEMMA_PROJECTION_FIXTURE").ok_or("fixture_missing")?);
    let output = PathBuf::from(std::env::var_os("STARK_GEMMA_PROJECTION_OUTPUT").ok_or("output_missing")?);
    let receipt_bytes = fs::read(fixture.join("receipt.json")).map_err(|_| "receipt_read")?;
    let receipt: serde_json::Value = serde_json::from_slice(&receipt_bytes).map_err(|_| "receipt_parse")?;
    if receipt["device"] != "gpu"
        || receipt["identity"]["mlx_lm_revision"] != "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"
        || receipt["identity"]["actual_projection_rows"] != 1
        || receipt["identity"]["cached_last"] != true
    {
        return Err("source_identity");
    }
    let read = |name: &str| -> Result<Vec<f32>, &'static str> {
        let data = fs::read(fixture.join(format!("{name}.f32"))).map_err(|_| "tensor_read")?;
        let entry = receipt["operations"]
            .as_array()
            .ok_or("operations_missing")?
            .iter()
            .find(|e| e["operation"] == name)
            .ok_or("tensor_missing")?;
        if !data.len().is_multiple_of(4) || entry["sha256"] != format!("{:x}", Sha256::digest(&data)) {
            return Err("tensor_hash");
        }
        let values: Vec<_> = data.as_chunks::<4>().0.iter().map(|x| f32::from_le_bytes(*x)).collect();
        if values.iter().any(|&x| !x.is_finite() || bf16::from_f32(x).to_f32() != x) {
            return Err("tensor_dtype");
        }
        Ok(values)
    };
    let context = <Metal as Backend>::Context::new().map_err(|_| "context")?;
    let q = read("actual_q_rope")?;
    let mut k = read("actual_attention_keys_native")?;
    let mut v = read("actual_attention_values_native")?;
    let expected = read("actual_attention_output")?;
    let absolute_position = receipt["identity"]["actual_position_offset"].as_u64().ok_or("prefix_missing")? as usize;
    let head_dim = q.len() / 8;
    if !matches!(head_dim, 256 | 512) {
        return Err("source_head_dimension");
    }
    let mut sequence = k.len() / head_dim;
    if !matches!(head_dim, 256 | 512)
        || q.len() != 8 * head_dim
        || k.len() != sequence * head_dim
        || v.len() != k.len()
        || expected.len() != q.len()
    {
        return Err("source_shape");
    }
    let mut layout = "source-physical";
    let mut cache_view = KVCacheView::full(sequence as u32 - 1);
    let mut ring = false;
    let mut window = None;
    let mut native_cache_identity = None;
    if let Some(path) = std::env::var_os("STARK_GEMMA_NATIVE_CACHE_PATH") {
        let path = PathBuf::from(path);
        let metadata_bytes = fs::read(path.join("cache-view.json")).map_err(|_| "cache_metadata")?;
        let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes).map_err(|_| "cache_json")?;
        if head_dim != 256
            || absolute_position != 512
            || metadata["layer_index"] != 8
            || metadata["step"] != 1
            || metadata["rows"] != 513
            || metadata["prefix_rows"] != 512
            || metadata["ring_offset"] != 0
            || metadata["ring_length"] != 512
            || metadata["element_dim"] != 256
        {
            return Err("cache_geometry");
        }
        let parent = path.parent().ok_or("cache_parent")?;
        let trajectory_bytes = fs::read(parent.join("trajectory-native.json")).map_err(|_| "cache_receipt")?;
        let trajectory: serde_json::Value =
            serde_json::from_slice(&trajectory_bytes).map_err(|_| "cache_receipt_json")?;
        let entries = trajectory["modes"][1]["steps"][1]["components"].as_array().ok_or("cache_components")?;
        let read_native = |name: &str| -> Result<Vec<f32>, &'static str> {
            let bytes = fs::read(path.join(format!("{name}.f32"))).map_err(|_| "cache_tensor")?;
            if bytes.len() != 513 * 256 * 4
                || !entries
                    .iter()
                    .any(|e| e["component"] == name && e["sha256"] == format!("{:x}", Sha256::digest(&bytes)))
            {
                return Err("cache_hash");
            }
            let values: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|x| f32::from_le_bytes(*x)).collect();
            if values.iter().any(|&x| !x.is_finite() || bf16::from_f32(x).to_f32() != x) {
                return Err("cache_dtype");
            }
            Ok(values)
        };
        k = read_native("native_cache_keys")?;
        v = read_native("native_cache_values")?;
        native_cache_identity = Some(format!("{:x}", Sha256::digest(&trajectory_bytes)));
        if std::env::var_os("STARK_GEMMA_CACHE_COMPACT_SOURCE_ORDER").is_some() {
            let compact =
                |a: &[f32]| -> Vec<f32> { a[512 * 256..].iter().chain(a[256..512 * 256].iter()).copied().collect() };
            k = compact(&k);
            v = compact(&v);
            sequence = 512;
            layout = "native-compacted-source-order";
        } else {
            sequence = 513;
            cache_view = KVCacheView::ring(512, 0);
            ring = true;
            window = Some(512);
            layout = "native-physical-prefix-plus-suffix";
        }
    }
    let to_buffer = |values: &[f32]| {
        crate::tests::helpers::create_buffer_with_data::<Metal, bf16>(
            &context,
            &values.iter().copied().map(bf16::from_f32).collect::<Vec<_>>(),
        )
    };
    let q = to_buffer(&q);
    let k = to_buffer(&k);
    let v = to_buffer(&v);
    let config = AttentionKernelConfig {
        head_dim: head_dim as u32,
        num_groups: 1,
        num_q_heads: 8,
        has_sinks: false,
        is_kv_cache_ring: ring,
        is_causal: true,
        sliding_window_size: window,
        scale: Some(1.0),
        data_type: DataType::BF16,
        materialize_prefill_scores: true,
        source_bf16_prefill: false,
    };
    let mut commands = context.create_command_buffer(None, None).map_err(|_| "commands")?;
    let kernel =
        <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(&context, config).map_err(|_| "kernel")?;
    let native = kernel
        .encode(
            AttentionArguments {
                queries: &q,
                keys: &k,
                values: &v,
                suffix_length: 1,
                trie: None::<&<Metal as Backend>::ConstantBuffer>,
                sinks: None,
                cache: cache_view,
            },
            &mut commands,
        )
        .map_err(|_| "encode")?;
    let mut copy = context.create_buffer(native.size()).map_err(|_| "copy")?;
    commands.encode_copy(&native, &mut copy);
    commands.end_encoding().submit().wait_until_completed().map_err(|_| "execute")?;
    let native: Vec<_> = copy.copyout::<bf16>().into_iter().map(|x| x.to_f32()).collect();
    let error: Vec<_> = native.iter().zip(&expected).map(|(&a, &b)| f64::from(a) - f64::from(b)).collect();
    let bytes: Vec<_> = native.iter().flat_map(|x| x.to_le_bytes()).collect();
    let report = serde_json::json!({"backend":"metal","acceptance":"diagnostic-only","source_layer":receipt["identity"]["actual_layer_index"],
        "prefix_tokens":cache_view.prefix_len(),"source_absolute_position":absolute_position,"layout":layout,"attention_sequence_length":sequence,"native_cache_receipt_sha256":native_cache_identity,"head_dim":head_dim,"elements":native.len(),"different":error.iter().filter(|x|**x!=0.0).count(),
        "rmse":(error.iter().map(|x|x*x).sum::<f64>()/error.len() as f64).sqrt(),"max_abs":error.iter().map(|x|x.abs()).fold(0.0,f64::max),
        "native_sha256":format!("{:x}",Sha256::digest(&bytes)),"fixture_receipt_sha256":format!("{:x}",Sha256::digest(&receipt_bytes))});
    private_write(&output, &serde_json::to_vec(&report).map_err(|_| "report_encode")?).map_err(|_| "report_write")?;
    Ok(())
}

#[stark_test]
#[ignore]
fn gemma4_private_cached_vector_attention_probe() {
    let result = cached_vector_attention_probe();
    assert!(result.is_ok(), "private cached vector attention probe failed: {}", result.err().unwrap_or("unknown"));
}
