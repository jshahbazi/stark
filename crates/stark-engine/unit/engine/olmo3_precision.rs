//! Explicitly requested, private model-bound operator tracing. Never run by CI.

use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::PathBuf,
};

use half::bf16;
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use crate::{
    backends::{
        common::{
            Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable,
            CommandBufferPending, Context,
            gpu_types::trie::TrieNode,
            kernel::{AttentionArguments, AttentionKernel, AttentionKernelConfig, Kernels, TensorAddScaleKernel},
        },
        metal::Metal,
    },
    config::decoder::DecoderConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        decoder::Decoder,
        linear::Linear,
        mixer::attention::{KVCacheView, rope::PrecalculatedRoPE},
        transformer_layer::TransformerLayer,
    },
    parameters::ParameterLoader,
    tests::private_output::{create_private_directory, private_write},
};

fn prefill_trace() -> Result<(), &'static str> {
    let model_path = PathBuf::from(std::env::var_os("STARK_OLMO_TRACE_MODEL").ok_or("trace_model_missing")?);
    let fixture_path = PathBuf::from(std::env::var_os("STARK_OLMO_TRACE_FIXTURE").ok_or("trace_fixture_missing")?);
    let output_path = PathBuf::from(std::env::var_os("STARK_OLMO_TRACE_OUTPUT").ok_or("trace_output_missing")?);
    let raw: serde_json::Value =
        serde_json::from_slice(&fs::read(model_path.join("config.json")).map_err(|_| "config_read_failed")?)
            .map_err(|_| "config_invalid")?;
    let config: DecoderConfig =
        serde_json::from_value(raw["decoder_config"].clone()).map_err(|_| "decoder_config_invalid")?;
    let fixture: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture_path.join("cases.json")).map_err(|_| "fixture_read_failed")?)
            .map_err(|_| "fixture_invalid")?;
    let case_index = std::env::var("STARK_OLMO_TRACE_CASE_INDEX")
        .ok()
        .map(|value| value.parse::<usize>().map_err(|_| "case_index_invalid"))
        .transpose()?
        .unwrap_or(0);
    let mut tokens: Vec<u32> =
        serde_json::from_value(fixture["cases"][case_index]["tokens"].clone()).map_err(|_| "tokens_invalid")?;
    let prefill = std::env::var_os("STARK_OLMO_TRACE_PREFILL").is_some();
    let cached_last = std::env::var_os("STARK_OLMO_TRACE_CACHED_LAST").is_some();
    if prefill && cached_last {
        return Err("incompatible_trace_modes");
    }
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
        || config.transformer_config.layer_configs.len() != 32
        || config.ple_model_config.is_some()
    {
        return Err("trace_requires_bounded_olmo3_prefix");
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
        context.create_command_buffer(Some("private OLMo precision trace"), None).map_err(|_| "commands_failed")?;
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
    let layers: Vec<u32> = (0..32).collect();
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
    create_private_directory(&output_path)?;
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
    )?;
    Ok(())
}

fn trace() -> Result<(), &'static str> {
    if std::env::var_os("STARK_OLMO_TRACE_PREFILL").is_some()
        || std::env::var_os("STARK_OLMO_TRACE_CACHED_LAST").is_some()
    {
        return prefill_trace();
    }
    let model_path = PathBuf::from(std::env::var_os("STARK_OLMO_TRACE_MODEL").ok_or("trace_model_missing")?);
    let fixture_path = PathBuf::from(std::env::var_os("STARK_OLMO_TRACE_FIXTURE").ok_or("trace_fixture_missing")?);
    let output_path = PathBuf::from(std::env::var_os("STARK_OLMO_TRACE_OUTPUT").ok_or("trace_output_missing")?);
    let raw: serde_json::Value =
        serde_json::from_slice(&fs::read(model_path.join("config.json")).map_err(|_| "config_read_failed")?)
            .map_err(|_| "config_invalid")?;
    let config: DecoderConfig =
        serde_json::from_value(raw["decoder_config"].clone()).map_err(|_| "decoder_config_invalid")?;
    let fixture: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture_path.join("cases.json")).map_err(|_| "fixture_read_failed")?)
            .map_err(|_| "fixture_invalid")?;
    let tokens: Vec<u32> =
        serde_json::from_value(fixture["cases"][0]["tokens"].clone()).map_err(|_| "tokens_invalid")?;
    if tokens.len() != 1
        || config.transformer_config.layer_configs.len() != 32
        || config.transformer_config.layer_configs.iter().any(|layer| !layer.post_norm)
    {
        return Err("trace_requires_one_token_olmo3");
    }
    let context = <Metal as Backend>::Context::new().map_err(|_| "context_failed")?;
    let weights = File::open(model_path.join("model.safetensors")).map_err(|_| "weights_read_failed")?;
    let loader = ParameterLoader::<Metal>::new(&weights, &context).map_err(|_| "loader_failed")?;
    let decoder = Decoder::<Metal>::new(&context, &config, &loader.tree().subtree("decoder"), DataType::BF16)
        .map_err(|_| "decoder_failed")?;
    loader.tree().assert_all_tensors_validated().map_err(|_| "unvalidated_tensor")?;
    let mut state = decoder.create_empty_state(Some(1024), &context).map_err(|_| "state_failed")?;
    state.prepare(0, 1).map_err(|_| "state_prepare_failed")?;
    let mut commands =
        context.create_command_buffer(Some("private OLMo precision trace"), None).map_err(|_| "commands_failed")?;
    let token_ids = commands.allocate_constant_from_slice(&tokens).map_err(|_| "token_buffer_failed")?;
    let embedding = decoder.embedding().encode_lookup(&token_ids, 1, &mut commands).map_err(|_| "embedding_failed")?;
    let nodes = [TrieNode {
        trie_start: 0,
        trie_end: 1,
        height: 0,
    }];
    let topology = BatchTopology::new(&nodes, true);
    let layers: Vec<u32> = (0..32).collect();
    let output = decoder
        .encode(&token_ids, &topology, Some((0..1).into()), Some(&layers), &mut state, &mut commands)
        .map_err(|_| "decode_failed")?;
    let layer_index =
        std::env::var("STARK_OLMO_TRACE_LAYER").ok().and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);
    if layer_index >= 32 {
        return Err("trace_layer_invalid");
    }
    let input = if layer_index == 0 {
        &embedding
    } else {
        &output.hidden_features.as_ref().ok_or("hidden_missing")?[layer_index - 1]
    };
    let tree = loader.tree().subtree(&format!("decoder.transformer.layers.{layer_index}"));
    let layer = TransformerLayer::<Metal>::new(
        &context,
        config.transformer_config.model_dim,
        config.transformer_config.hidden_dim,
        32,
        &config.transformer_config.layer_configs[layer_index],
        layer_index as u32,
        &tree,
        DataType::BF16,
    )
    .map_err(|_| "initial_layer_failed")?;
    let width = config.transformer_config.model_dim;
    let rope = PrecalculatedRoPE::<Metal>::precalculate(
        config.transformer_config.layer_configs[layer_index].rope_config.as_ref().ok_or("initial_rope_missing")?,
        &[0],
        &mut commands,
    )
    .map_err(|_| "initial_rope_failed")?;
    let mut details = Vec::new();
    let duplicate =
        |source: &<Metal as Backend>::ScratchBuffer,
         commands: &mut <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding| {
            let mut copy = commands.allocate_scratch(source.size())?;
            commands.encode_copy(source, &mut copy);
            Ok::<_, <Metal as Backend>::Error>(copy)
        };
    let projection = <dyn Linear<Metal>>::new(
        width,
        [3 * width],
        false,
        &context,
        DataType::BF16,
        &tree.subtree("mixer.qkvg_projection"),
    )
    .map_err(|_| "initial_projection_failed")?;
    details.push((
        "initial-qkv".to_owned(),
        projection
            .encode(duplicate(input, &mut commands).map_err(|_| "initial_copy_failed")?, 1, &mut commands)
            .map_err(|_| "initial_projection_failed")?,
    ));
    let attention = layer
        .mixer
        .encode(
            duplicate(input, &mut commands).map_err(|_| "initial_copy_failed")?,
            Some(&rope),
            &topology,
            None,
            &mut commands,
        )
        .map_err(|_| "initial_attention_failed")?;
    let normalized = layer
        .post_mixer_norm
        .as_ref()
        .ok_or("initial_norm_missing")?
        .encode(&attention, 0, 1, None::<&mut <Metal as Backend>::ScratchBuffer>, &mut commands)
        .map_err(|_| "initial_norm_failed")?;
    let add = <<Metal as Backend>::Kernels as Kernels>::TensorAddScaleKernel::new(&context, DataType::BF16, false)
        .map_err(|_| "initial_add_failed")?;
    let mut pre_mlp = commands.allocate_scratch(input.size()).map_err(|_| "initial_copy_failed")?;
    add.encode(Some(input), &normalized, &mut pre_mlp, width, width, 1.0, &mut commands);
    details.push(("initial-attnout".to_owned(), attention));
    details.push(("initial-postattn".to_owned(), normalized));
    let up = <dyn Linear<Metal>>::new(
        width,
        [2 * config.transformer_config.hidden_dim],
        false,
        &context,
        DataType::BF16,
        &tree.subtree("mlp.up_projection"),
    )
    .map_err(|_| "initial_up_failed")?;
    details.push((
        "initial-up".to_owned(),
        up.encode(duplicate(&pre_mlp, &mut commands).map_err(|_| "initial_copy_failed")?, 1, &mut commands)
            .map_err(|_| "initial_up_failed")?,
    ));
    let down = layer
        .mlp
        .encode(duplicate(&pre_mlp, &mut commands).map_err(|_| "initial_copy_failed")?, 1, &mut commands)
        .map_err(|_| "initial_mlp_failed")?;
    let post_mlp = layer
        .post_mlp_norm
        .as_ref()
        .ok_or("initial_norm_missing")?
        .encode(&down, 0, 1, None::<&mut <Metal as Backend>::ScratchBuffer>, &mut commands)
        .map_err(|_| "initial_norm_failed")?;
    let mut result = commands.allocate_scratch(input.size()).map_err(|_| "initial_copy_failed")?;
    add.encode(Some(&pre_mlp), &post_mlp, &mut result, width, width, 1.0, &mut commands);
    details.push(("initial-premlp".to_owned(), pre_mlp));
    details.push(("initial-down".to_owned(), down));
    details.push(("initial-postmlp".to_owned(), post_mlp));
    details.push(("initial-result".to_owned(), result));
    let mut sources = vec![("embedding".to_owned(), embedding)];
    for (index, hidden) in output.hidden_features.ok_or("hidden_missing")?.into_vec().into_iter().enumerate() {
        sources.push((format!("layer-{index}"), hidden));
    }
    sources.push(("final-hidden".to_owned(), output.final_hidden.ok_or("final_hidden_missing")?));
    sources.push(("logits".to_owned(), output.logits.ok_or("logits_missing")?));
    sources.extend(details);
    let mut readbacks = Vec::new();
    for (name, source) in sources {
        let mut destination = context.create_buffer(source.size()).map_err(|_| "readback_failed")?;
        commands.encode_copy(&source, &mut destination);
        readbacks.push((name, destination));
    }
    drop(token_ids);
    drop(rope);
    commands.end_encoding().submit().wait_until_completed().map_err(|_| "execution_failed")?;
    create_private_directory(&output_path)?;
    let mut rows = Vec::new();
    for (name, readback) in readbacks {
        let values: Vec<bf16> = readback.copyout();
        if values.iter().any(|value| !value.is_finite()) {
            return Err("trace_nonfinite");
        }
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_f32().to_le_bytes()).collect();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(output_path.join(format!("{name}.f32")))
            .map_err(|_| "trace_file_create_failed")?;
        file.write_all(&bytes).map_err(|_| "trace_write_failed")?;
        rows.push(serde_json::json!({"component":name,"elements":values.len(),"sha256":digest}));
    }
    let report = serde_json::json!({"backend":"metal","precision":"bf16","input_tokens":tokens.len(),"operator_layer":layer_index,"components":rows});
    private_write(&output_path.join("report.json"), &serde_json::to_vec(&report).map_err(|_| "report_encode_failed")?)?;
    Ok(())
}

#[stark_test]
#[ignore] // Requires explicitly selected private model, fixture, and fresh output directory.
fn olmo3_private_precision_trace() {
    let result = trace();
    assert!(result.is_ok(), "private OLMo precision trace failed: {}", result.err().unwrap_or("unknown"));
}

fn attention_fixture_probe() -> Result<(), &'static str> {
    let root = PathBuf::from(std::env::var_os("STARK_NATIVE_ATTENTION_FIXTURE").ok_or("attention_fixture_missing")?);
    let output = PathBuf::from(std::env::var_os("STARK_NATIVE_ATTENTION_OUTPUT").ok_or("attention_output_missing")?);
    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("receipt.json")).map_err(|_| "attention_receipt_read_failed")?)
            .map_err(|_| "attention_receipt_invalid")?;
    let geometry = &receipt["trace_geometry"];
    let rows = geometry["input_tokens"].as_u64().ok_or("attention_geometry_invalid")? as usize;
    let heads = geometry["attention_heads"].as_u64().ok_or("attention_geometry_invalid")? as usize;
    let groups = geometry["attention_groups"].as_u64().ok_or("attention_geometry_invalid")? as usize;
    let head_dim = geometry["attention_head_dim"].as_u64().ok_or("attention_geometry_invalid")? as usize;
    let window = geometry["sliding_window"].as_u64().ok_or("attention_geometry_invalid")? as u32;
    let scale = geometry["attention_scale"].as_f64().ok_or("attention_geometry_invalid")? as f32;
    let kv_rows = geometry["kv_tokens"].as_u64().map_or(rows, |n| n as usize);
    if kv_rows < rows || kv_rows > 1024 {
        return Err("attention_kv_bounds");
    }
    if !((rows == 511 && heads == 8 && groups == 1 && head_dim == 256 && window == 512 && scale == 1.0)
        || ((rows == 49 || rows == 1)
            && heads == 32
            && groups == 32
            && head_dim == 128
            && window >= 49
            && scale.is_finite()))
    {
        return Err("attention_geometry_unsupported");
    }
    let load = |name: &str, shape: &[usize]| -> Result<Vec<bf16>, &'static str> {
        let expected = receipt["bos_trace_components"]
            .as_array()
            .ok_or("attention_components_missing")?
            .iter()
            .find(|value| value["component"].as_str() == Some(name))
            .ok_or("attention_component_missing")?;
        let count: usize = shape.iter().product();
        let bytes = fs::read(root.join("trace-prefill").join(format!("{name}.f32")))
            .map_err(|_| "attention_component_read_failed")?;
        if bytes.len() != count * size_of::<f32>()
            || expected["shape"] != serde_json::json!(shape)
            || expected["elements"] != count
            || expected["sha256"].as_str() != Some(format!("{:x}", Sha256::digest(&bytes)).as_str())
        {
            return Err("attention_component_identity_failed");
        }
        bytes
            .chunks_exact(size_of::<f32>())
            .map(|bytes| {
                let value = f32::from_le_bytes(bytes.try_into().map_err(|_| "attention_value_invalid")?);
                let narrowed = bf16::from_f32(value);
                if !value.is_finite() || narrowed.to_f32().to_bits() != value.to_bits() {
                    return Err("attention_component_not_bf16");
                }
                Ok(narrowed)
            })
            .collect()
    };
    let layer = std::env::var("STARK_NATIVE_ATTENTION_LAYER")
        .ok()
        .map(|value| value.parse::<u32>().map_err(|_| "attention_layer_invalid"))
        .transpose()?
        .unwrap_or(0);
    if layer >= 32 {
        return Err("attention_layer_unsupported");
    }
    let is_full_attention =
        head_dim == 128 && receipt["geometry"]["layer_types"][layer as usize].as_str() == Some("full_attention");
    let sliding_window = (!is_full_attention).then_some(window);
    let q = load(&format!("attention-query-{layer}"), &[heads, rows, head_dim])?;
    let k = load(&format!("attention-key-native-{layer}"), &[kv_rows, groups, head_dim])?;
    let v = load(&format!("attention-value-native-{layer}"), &[kv_rows, groups, head_dim])?;
    let reference = load(&format!("attention-output-native-{layer}"), &[rows, heads, head_dim])?;
    let context = <Metal as Backend>::Context::new().map_err(|_| "attention_context_failed")?;
    let queries = crate::tests::helpers::create_buffer_with_data::<Metal, bf16>(&context, &q);
    let keys = crate::tests::helpers::create_buffer_with_data::<Metal, bf16>(&context, &k);
    let values = crate::tests::helpers::create_buffer_with_data::<Metal, bf16>(&context, &v);
    let path = std::env::var("STARK_NATIVE_ATTENTION_PATH").unwrap_or_else(|_| "auto".into());
    if !matches!(
        path.as_str(),
        "auto" | "unfused-bf16" | "materialized" | "mxu" | "source-scale-mxu" | "source-profile"
    ) {
        return Err("attention_path_unsupported");
    }
    let config = AttentionKernelConfig {
        head_dim: head_dim as u32,
        num_groups: groups as u32,
        num_q_heads: heads as u32,
        has_sinks: false,
        is_kv_cache_ring: !is_full_attention,
        is_causal: true,
        sliding_window_size: sliding_window,
        scale: Some(scale),
        data_type: DataType::BF16,
        materialize_prefill_scores: path == "materialized",
        source_bf16_prefill: path == "source-profile",
    };
    let mut commands = context
        .create_command_buffer(Some("private identical-input attention probe"), None)
        .map_err(|_| "attention_commands_failed")?;
    let arguments = AttentionArguments {
        queries: &queries,
        keys: &keys,
        values: &values,
        suffix_length: rows as u32,
        trie: None::<&<Metal as Backend>::GlobalBuffer>,
        sinks: None,
        cache: if is_full_attention {
            KVCacheView::full((kv_rows - rows) as u32)
        } else {
            KVCacheView::ring((kv_rows - rows) as u32, 0)
        },
    };
    let native = if path == "unfused-bf16" {
        <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::encode_unfused_prefill_for_test(
            &context,
            config,
            arguments,
            &mut commands,
        )
    } else if path == "source-scale-mxu" {
        <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::encode_source_scale_prefill_for_test(
            config,
            arguments,
            &mut commands,
        )
    } else if path == "mxu" {
        <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::encode_mxu_prefill_for_test(
            config,
            arguments,
            &mut commands,
        )
    } else {
        <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(&context, config)
            .map_err(|_| "attention_kernel_failed")?
            .encode(arguments, &mut commands)
    }
    .map_err(|_| "attention_encode_failed")?;
    let mut readback = context.create_buffer(native.size()).map_err(|_| "attention_readback_failed")?;
    commands.encode_copy(&native, &mut readback);
    commands.end_encoding().submit().wait_until_completed().map_err(|_| "attention_execution_failed")?;
    let native: Vec<bf16> = readback.copyout();
    if native.len() != reference.len() || native.iter().any(|value| !value.is_finite()) {
        return Err("attention_output_invalid");
    }
    let mut unequal = 0usize;
    let mut sum_square = 0.0f64;
    let mut max_abs = 0.0f64;
    for (actual, expected) in native.iter().zip(&reference) {
        let difference = f64::from(actual.to_f32()) - f64::from(expected.to_f32());
        unequal += usize::from(actual.to_bits() != expected.to_bits());
        sum_square += difference * difference;
        max_abs = max_abs.max(difference.abs());
    }
    let hash = |values: &[bf16]| {
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_f32().to_le_bytes()).collect();
        format!("{:x}", Sha256::digest(&bytes))
    };
    let report = serde_json::json!({
        "backend":"metal", "path":path,"layer":layer,"acceptance":"diagnostic-no-numerical-gate",
        "fixture_receipt_sha256": format!("{:x}",Sha256::digest(fs::read(root.join("receipt.json"))
            .map_err(|_| "attention_receipt_read_failed")?)),
        "input_tokens":rows,"kv_tokens":kv_rows,"heads":heads,"groups":groups,"head_dim":head_dim,"sliding_window":sliding_window,
        "causal":true,"scale":scale,"cache":if is_full_attention {"full"} else {"ring"},
        "elements":native.len(),"unequal_elements":unequal,"exact":unequal==0,
        "max_abs_error":max_abs,"rmse":(sum_square/native.len() as f64).sqrt(),
        "reference_sha256":hash(&reference),"native_sha256":hash(&native)
    });
    private_write(&output, &serde_json::to_vec(&report).map_err(|_| "attention_report_encode_failed")?)?;
    Ok(())
}

#[stark_test]
#[ignore] // Requires an explicit private, source-bound same-Q/K/V fixture and fresh report path.
fn native_private_attention_fixture_probe() {
    let result = attention_fixture_probe();
    assert!(result.is_ok(), "private attention probe failed: {}", result.err().unwrap_or("unknown"));
}
