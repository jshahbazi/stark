use std::{collections::BTreeMap, fs, path::PathBuf};

use serde_json::json;
use stark_engine_macros::stark_test;

use super::*;

type Tensors = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

fn config() -> Value {
    json!({"model_type":"ricursive", "architectures":["RicursiveForCausalLM"],
        "hidden_act":"silu", "tie_word_embeddings":false, "hidden_size":4,
        "intermediate_size":6, "num_attention_heads":4, "num_key_value_heads":2,
        "head_dim":2, "num_hidden_layers":3, "vocab_size":5,
        "max_position_embeddings":6, "sliding_window":2,
        "layer_types":["sliding_attention","full_attention","sliding_attention"],
        "rms_norm_eps":1e-6, "rope_theta":10000.0, "embed_scale":3.0,
        "ffn_clamp_threshold":0.25, "use_gated_attention":true,
        "gate_full_precision":true, "use_valres":true})
}

#[stark_test]
fn request_memory_covers_cache_rounding_overlap_and_rejects_excessive_context() {
    let mut c = Config::parse(&config()).unwrap();
    c.context = 65_536;
    c.window = 8_192;
    assert_eq!(c.graph_capacity(1, false), Some(16));
    assert_eq!(c.graph_capacity(16_384, false), Some(16_384));
    assert_eq!(c.graph_capacity(16_385, false), Some(20_480));
    assert_eq!(c.graph_capacity(16_385, true), Some(8_192));
    let per_row = (2 * c.kv_heads * c.dim + c.heads) * 4;
    let target_and_overlap = (8_192 + 20_480 + 8_192 + 20_480) * per_row;
    assert!(c.graph_request_memory_bytes(16_385).unwrap() >= target_and_overlap + 512 * 1024 * 1024);
    assert_eq!(c.graph_request_memory_bytes(0), None);
    assert_eq!(c.graph_request_memory_bytes(65_537), None);
    c.context = 16_777_216;
    c.kv_heads = 512;
    c.dim = 512;
    assert_eq!(c.graph_request_memory_bytes(c.context), None);
}

#[stark_test]
fn request_admission_releases_previous_ripple_state_and_preserves_reset_outputs() {
    let fixture = Fixture::new(&tensors(), &config());
    let mut model = fixture.load(&config()).unwrap();
    assert!(model.request_memory_bytes(6).is_some());
    assert_eq!(model.request_memory_bytes(0), None);
    assert_eq!(model.request_memory_bytes(7), None);
    let expected = model.step(1).unwrap();
    assert!(model.prepare_request_memory());
    assert_eq!(model.position, 0);
    assert!(model.states.iter().all(|state| state.keys.is_empty() && state.values.is_empty()));
    #[cfg(feature = "metal")]
    assert!(model.graph.is_none());
    assert!(!model.prepare_request_memory());
    assert_eq!(model.step(1).unwrap(), expected);
}

fn tensors() -> Tensors {
    let c = Config::parse(&config()).unwrap();
    let mut tensors = Tensors::new();
    let mut add = |name: String, shape: Vec<usize>| {
        let seed = name.bytes().map(usize::from).sum::<usize>();
        let count = shape.iter().product();
        let values = (0..count)
            .map(|i| {
                let v = ((i * 17 + seed * 7) % 53) as f32 - 26.0;
                if name.ends_with("norm.weight") {
                    1.0 + v * 0.008
                } else if name.contains(".mlp.gate_") || name.contains(".mlp.up_") {
                    v * 0.8
                } else {
                    v * 0.015
                }
            })
            .collect();
        tensors.insert(name, (shape, values));
    };
    add("model.embed_tokens.weight".into(), vec![c.vocab, c.hidden]);
    add("model.norm.weight".into(), vec![c.hidden]);
    add("lm_head.weight".into(), vec![c.vocab, c.hidden]);
    for i in 0..c.sliding.len() {
        let p = format!("model.layers.{i}");
        for n in
            ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]
        {
            add(format!("{p}.{n}.weight"), vec![c.hidden]);
        }
        for n in ["q", "k"] {
            add(format!("{p}.self_attn.{n}_norm.weight"), vec![c.dim]);
        }
        for n in ["q", "g"] {
            add(format!("{p}.self_attn.{n}_proj.weight"), vec![c.heads * c.dim, c.hidden]);
        }
        for n in ["k", "v"] {
            add(format!("{p}.self_attn.{n}_proj.weight"), vec![c.kv_heads * c.dim, c.hidden]);
        }
        add(format!("{p}.self_attn.o_proj.weight"), vec![c.hidden, c.heads * c.dim]);
        add(format!("{p}.self_attn.value_res_lambda"), vec![1]);
        for n in ["gate", "up"] {
            add(format!("{p}.mlp.{n}_proj.weight"), vec![c.intermediate, c.hidden]);
        }
        add(format!("{p}.mlp.down_proj.weight"), vec![c.hidden, c.intermediate]);
    }
    tensors
}

struct Fixture(PathBuf);

impl Fixture {
    fn new(
        tensors: &Tensors,
        config: &Value,
    ) -> Self {
        let path = std::env::temp_dir().join(format!(
            "stark-ripple-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        let mut metadata = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, (shape, values)) in tensors {
            let start = payload.len();
            payload.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            metadata.insert(name.clone(), json!({"dtype":"F32","shape":shape,"data_offsets":[start,payload.len()]}));
        }
        let mut header = serde_json::to_vec(&metadata).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        fs::write(path.join("model.safetensors"), bytes).unwrap();
        fs::write(path.join("config.json"), serde_json::to_vec(config).unwrap()).unwrap();
        Self(path)
    }

    fn load(
        &self,
        config: &Value,
    ) -> Result<RippleModel> {
        RippleModel::new(Weights::open(&self.0)?, config)
    }

    fn gguf(
        &self,
        tensors: &Tensors,
        alternate_residual: bool,
    ) {
        self.gguf_with_geometry(tensors, alternate_residual, None);
    }

    fn gguf_with_geometry(
        &self,
        tensors: &Tensors,
        alternate_residual: bool,
        blocks: Option<u32>,
    ) {
        let string = |bytes: &mut Vec<u8>, s: &str| {
            bytes.extend((s.len() as u64).to_le_bytes());
            bytes.extend(s.as_bytes());
        };
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend((tensors.len() as u64).to_le_bytes());
        bytes.extend((1 + u64::from(blocks.is_some())).to_le_bytes());
        string(&mut bytes, "general.architecture");
        bytes.extend(8u32.to_le_bytes());
        string(&mut bytes, "ricursive");
        if let Some(blocks) = blocks {
            string(&mut bytes, "ricursive.block_count");
            bytes.extend(4u32.to_le_bytes());
            bytes.extend(blocks.to_le_bytes());
        }
        let mut payload = Vec::new();
        for (name, (shape, values)) in tensors {
            string(&mut bytes, &gguf_name(name, alternate_residual));
            bytes.extend((shape.len() as u32).to_le_bytes());
            for dim in shape.iter().rev() {
                bytes.extend((*dim as u64).to_le_bytes());
            }
            bytes.extend(0u32.to_le_bytes());
            bytes.extend((payload.len() as u64).to_le_bytes());
            payload.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            while !payload.len().is_multiple_of(32) {
                payload.push(0);
            }
        }
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        bytes.extend(payload);
        fs::remove_file(self.0.join("model.safetensors")).unwrap();
        fs::write(self.0.join("model.gguf"), bytes).unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn gguf_name(
    name: &str,
    alternate_residual: bool,
) -> String {
    if name.starts_with("blk.") {
        return name.to_owned();
    }
    match name {
        "model.embed_tokens.weight" => return "token_embd.weight".into(),
        "model.norm.weight" => return "output_norm.weight".into(),
        "lm_head.weight" => return "output.weight".into(),
        _ => {},
    }
    let rest = name.strip_prefix("model.layers.").unwrap();
    let (layer, kind) = rest.split_once('.').unwrap();
    let gg = match kind {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
        "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.g_proj.weight" => "attn_gate.weight",
        "self_attn.value_res_lambda" => {
            if alternate_residual {
                "attn_value_res.weight"
            } else {
                "attn_vres_lambda.weight"
            }
        },
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => panic!("unexpected synthetic tensor"),
    };
    format!("blk.{layer}.{gg}")
}

// Independent f64 forward over dense source tensors and untrimmed histories.
// It does not call the native norm, rotary, gating or attention helpers.
fn reference(
    tensors: &Tensors,
    sequence: &[u32],
) -> Vec<Vec<f64>> {
    reference_options(tensors, sequence, true, true, Some(0.25))
}

fn reference_options(
    tensors: &Tensors,
    sequence: &[u32],
    gated: bool,
    valres: bool,
    clamp: Option<f64>,
) -> Vec<Vec<f64>> {
    let t = |name: &str| tensors[name].1.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let matrix = |name: &str, x: &[f64]| {
        let (shape, values) = &tensors[name];
        values
            .chunks_exact(shape[1])
            .map(|row| row.iter().zip(x).map(|(&w, &x)| f64::from(w) * x).sum())
            .collect::<Vec<_>>()
    };
    let norm = |name: &str, x: &[f64]| {
        let scale = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + f64::from(1e-6f32)).sqrt().recip();
        x.iter().zip(t(name)).map(|(&v, w)| v * scale * w).collect::<Vec<_>>()
    };
    let mut keys: Vec<Vec<Vec<f64>>> = vec![Vec::new(); 3];
    let mut values = keys.clone();
    let mut outputs = Vec::new();
    for (position, &token) in sequence.iter().enumerate() {
        let seed: Vec<_> = t("model.embed_tokens.weight")[token as usize * 4..token as usize * 4 + 4]
            .iter()
            .map(|v| v * 3.0)
            .collect();
        let mut x = seed.clone();
        for layer in 0..3 {
            let p = format!("model.layers.{layer}");
            let n = norm(&format!("{p}.input_layernorm.weight"), &x);
            let a = format!("{p}.self_attn");
            let mut q = matrix(&format!("{a}.q_proj.weight"), &n);
            let mut k = matrix(&format!("{a}.k_proj.weight"), &n);
            let blend = 1.0 / (1.0 + (-t(&format!("{a}.value_res_lambda"))[0]).exp());
            let v_input: Vec<_> = if valres {
                seed.iter().zip(&n).map(|(&s, &h)| (1.0 - blend) * s + blend * h).collect()
            } else {
                n.clone()
            };
            let v = matrix(&format!("{a}.v_proj.weight"), &v_input);
            for (heads, name) in [(&mut q, "q"), (&mut k, "k")] {
                for head in heads.chunks_exact_mut(2) {
                    let mut normalized = norm(&format!("{a}.{name}_norm.weight"), head);
                    if layer != 1 {
                        let (s, c) = (position as f64).sin_cos();
                        normalized = vec![normalized[0] * c - normalized[1] * s, normalized[0] * s + normalized[1] * c];
                    }
                    head.copy_from_slice(&normalized);
                }
            }
            keys[layer].push(k);
            values[layer].push(v);
            let start = if layer == 1 {
                0
            } else {
                (position + 1).saturating_sub(2)
            };
            let mut attn = vec![0.0; 8];
            for h in 0..4 {
                let kv = h / 2 * 2;
                let scores: Vec<_> = keys[layer][start..]
                    .iter()
                    .map(|k| (q[h * 2] * k[kv] + q[h * 2 + 1] * k[kv + 1]) / 2f64.sqrt())
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<_> = scores.iter().map(|v| (v - max).exp()).collect();
                let sum = exps.iter().sum::<f64>();
                for (weight, row) in exps.iter().zip(&values[layer][start..]) {
                    for j in 0..2 {
                        attn[h * 2 + j] += weight / sum * row[kv + j];
                    }
                }
            }
            if gated {
                let gates = matrix(&format!("{a}.g_proj.weight"), &n);
                for (value, gate) in attn.iter_mut().zip(gates) {
                    *value /= 1.0 + (-gate).exp();
                }
            }
            let attn = matrix(&format!("{a}.o_proj.weight"), &attn);
            let attn = norm(&format!("{p}.post_attention_layernorm.weight"), &attn);
            for j in 0..4 {
                x[j] += attn[j];
            }
            let n = norm(&format!("{p}.pre_feedforward_layernorm.weight"), &x);
            let m = format!("{p}.mlp");
            let g = matrix(&format!("{m}.gate_proj.weight"), &n);
            let u = matrix(&format!("{m}.up_proj.weight"), &n);
            let activated: Vec<_> = g
                .iter()
                .zip(u)
                .map(|(&g, u)| {
                    let (g, u) = clamp.map_or((g, u), |limit| (g.min(limit), u.clamp(-limit, limit)));
                    g / (1.0 + (-g).exp()) * u
                })
                .collect();
            let update = matrix(&format!("{m}.down_proj.weight"), &activated);
            let update = norm(&format!("{p}.post_feedforward_layernorm.weight"), &update);
            for j in 0..4 {
                x[j] += update[j];
            }
        }
        outputs.push(matrix("lm_head.weight", &norm("model.norm.weight", &x)));
    }
    outputs
}

fn close(
    actual: &[f32],
    expected: &[f64],
) {
    assert_eq!(actual.len(), expected.len());
    let maximum = actual.iter().zip(expected).map(|(&a, &b)| (f64::from(a) - b).abs()).fold(0.0, f64::max);
    assert!(maximum < 2e-5, "synthetic Ripple numerical mismatch: max abs error {maximum}");
}

#[stark_test]
fn full_forward_matches_independent_reference_past_sliding_window() {
    let tensors = tensors();
    let fixture = Fixture::new(&tensors, &config());
    let sequence = [0, 1, 2, 3, 4, 0];
    let expected = reference(&tensors, &sequence);
    let mut model = fixture.load(&config()).unwrap();
    #[cfg(feature = "metal")]
    {
        model.graph_enabled = false;
    }
    for (token, expected) in sequence.into_iter().zip(&expected) {
        close(&model.step(token).unwrap(), expected);
    }
    assert_eq!(model.states[0].keys.len(), 2);
    assert_eq!(model.states[1].keys.len(), 6);
    assert_eq!(model.states[2].keys.len(), 2);
    assert!(model.step(0).is_err());
    assert_eq!(model.position, 0);
    assert!(model.states.iter().all(|s| s.keys.is_empty() && s.values.is_empty()));
    close(&model.step(0).unwrap(), &expected[0]);
}

#[stark_test]
fn gguf_tensor_aliases_preserve_reference_semantics() {
    let tensors = tensors();
    let expected = reference(&tensors, &[0, 1, 2, 3]);
    for alternate_residual in [false, true] {
        let fixture = Fixture::new(&tensors, &config());
        fixture.gguf(&tensors, alternate_residual);
        let mut model = fixture.load(&config()).unwrap();
        for (token, expected) in [0, 1, 2, 3].into_iter().zip(&expected) {
            close(&model.step(token).unwrap(), expected);
        }
    }
}

#[stark_test]
fn prefill_reset_and_invalid_tokens_do_not_retain_partial_state() {
    let tensors = tensors();
    let fixture = Fixture::new(&tensors, &config());
    let expected = reference(&tensors, &[0, 1, 2, 3]);
    let mut model = fixture.load(&config()).unwrap();
    for token in [0, 1, 2] {
        model.ingest(token).unwrap();
    }
    close(&model.step(3).unwrap(), &expected[3]);
    assert!(model.step(5).is_err());
    assert_eq!(model.position, 0);
    close(&model.step(0).unwrap(), &expected[0]);
    model.reset();
    assert_eq!(model.position, 0);
    close(&model.step(0).unwrap(), &expected[0]);
}

#[stark_test]
fn unsupported_semantics_and_malformed_weights_fail_admission() {
    for (key, value) in [
        ("model_type", json!("llama")),
        ("hidden_act", json!("gelu")),
        ("tie_word_embeddings", json!(true)),
        ("attention_bias", json!(true)),
        ("use_valres", json!("true")),
        ("gate_full_precision", json!("true")),
        ("num_key_value_heads", json!(3)),
        ("head_dim", json!(3)),
        ("layer_types", json!(["sliding_attention"])),
        ("ffn_clamp_threshold", json!(-1)),
        ("hybrid_layer_pattern", json!([1, 1, 1])),
        ("rope_scaling", json!({"factor":2})),
    ] {
        let mut invalid = config();
        invalid[key] = value;
        assert!(Config::parse(&invalid).is_err(), "invalid synthetic Ripple configuration accepted");
    }
    let mut tensors = tensors();
    tensors.remove("model.layers.2.post_feedforward_layernorm.weight");
    let fixture = Fixture::new(&tensors, &config());
    assert!(fixture.load(&config()).is_err());
}

#[stark_test]
fn published_maximum_context_does_not_allocate_cache_at_load() {
    let mut config = config();
    config["max_position_embeddings"] = json!(2_097_152);
    let fixture = Fixture::new(&tensors(), &config);
    let model = fixture.load(&config).unwrap();
    assert!(model.states.iter().all(|s| s.keys.is_empty() && s.values.is_empty()));
    assert!(model.config.cache_bytes(1).unwrap() < 1024);
    assert!(model.config.cache_bytes(usize::MAX).is_err());
}

#[stark_test]
fn gguf_metadata_mismatch_and_ambiguous_residual_fail_admission() {
    let tensors = tensors();
    let fixture = Fixture::new(&tensors, &config());
    fixture.gguf_with_geometry(&tensors, false, Some(4));
    assert!(fixture.load(&config()).is_err());
    let mut ambiguous = tensors;
    let scalar = ambiguous["model.layers.0.self_attn.value_res_lambda"].clone();
    ambiguous.insert("blk.0.attn_value_res.weight".into(), scalar);
    let fixture = Fixture::new(&ambiguous, &config());
    fixture.gguf(&ambiguous, false);
    assert!(fixture.load(&config()).is_err());
}

#[cfg(feature = "metal")]
fn graph_growth_reference_checks(
    embedding_multiplier: f32,
    tolerance: f64,
) {
    let mut tensors = tensors();
    // Keep the extended four-channel synthetic trajectory away from a nearly
    // zero residual norm. Otherwise both F32 implementations amplify tiny
    // reduction differences against f64 after repeated normalized updates.
    // The original short fixture above retains its unchanged source tensors.
    for value in &mut tensors.get_mut("model.embed_tokens.weight").unwrap().1 {
        *value *= embedding_multiplier;
    }
    for (gated, valres, clamp) in [(true, true, 0.25), (false, false, 0.0)] {
        let mut c = config();
        c["max_position_embeddings"] = json!(40);
        c["use_gated_attention"] = json!(gated);
        c["use_valres"] = json!(valres);
        c["ffn_clamp_threshold"] = json!(clamp);
        let fixture = Fixture::new(&tensors, &c);
        let mut graph = fixture.load(&c).unwrap();
        assert!(graph.graph_admitted);
        let mut staged = fixture.load(&c).unwrap();
        staged.graph_enabled = false;
        let sequence: Vec<_> = (0..40).map(|index| index % 5).collect();
        let oracle = reference_options(
            &tensors,
            &sequence,
            gated,
            valres,
            if clamp == 0.0 {
                None
            } else {
                Some(clamp)
            },
        );
        let mut graph_error = 0.0f64;
        let mut staged_error = 0.0f64;
        let mut graph_staged_error = 0.0f64;
        let mut graph_squared_error = 0.0f64;
        let mut reference_squared = 0.0f64;
        let first = graph.step(0).unwrap();
        close(&first, &staged.step(0).unwrap().into_iter().map(f64::from).collect::<Vec<_>>());
        for index in 1..40 {
            let token = index % 5;
            if index % 4 == 0 {
                graph.ingest(token).unwrap();
                staged.ingest(token).unwrap();
            } else {
                let expected = staged.step(token).unwrap().into_iter().map(f64::from).collect::<Vec<_>>();
                let actual = graph.step(token).unwrap();
                for ((&actual, &expected), &reference) in actual.iter().zip(&expected).zip(&oracle[index as usize]) {
                    graph_error = graph_error.max((f64::from(actual) - reference).abs());
                    staged_error = staged_error.max((expected - reference).abs());
                    graph_staged_error = graph_staged_error.max((f64::from(actual) - expected).abs());
                    graph_squared_error += (f64::from(actual) - reference).powi(2);
                    reference_squared += reference.powi(2);
                }
            }
        }
        eprintln!(
            "Ripple graph aggregate: embedding_multiplier={embedding_multiplier} gated={gated} valres={valres} clamp={clamp} graph_f64_max={graph_error} staged_f64_max={staged_error} graph_staged_max={graph_staged_error} graph_f64_relative_l2={}",
            (graph_squared_error / reference_squared).sqrt()
        );
        assert!(graph_error < tolerance && staged_error < tolerance && graph_staged_error < tolerance);
        assert!(graph.step(0).is_err());
        assert_eq!(graph.position, 0);
        assert!(graph.graph.is_none());
        assert_eq!(graph.step(0).unwrap(), first);
        graph.reset();
        assert!(graph.graph.is_none());
        assert_eq!(graph.step(0).unwrap(), first);
        assert!(graph.ingest(u32::MAX).is_err());
        assert!(graph.graph.is_none());
        assert_eq!(graph.step(0).unwrap(), first);
    }
}

#[cfg(feature = "metal")]
#[stark_test]
fn device_graph_matches_staged_forward_through_cache_growth_and_ring_eviction() {
    graph_growth_reference_checks(10.0, 2e-5);
}

#[cfg(feature = "metal")]
#[stark_test]
fn device_graph_and_staged_bound_ill_conditioned_trajectory_against_f64() {
    // This unchanged four-channel fixture can nearly cancel its residual
    // after repeated norm-scaled updates. Both F32 paths accumulate larger
    // error against the independent f64 oracle; keep that evidence separate
    // from the stable strict comparison and unchanged short source fixture.
    graph_growth_reference_checks(1.0, 0.002);
}

#[cfg(feature = "metal")]
#[stark_test]
fn device_graph_rejects_nonfinite_projections_before_clipping_can_hide_them() {
    for poison in [f32::NAN, f32::INFINITY] {
        let mut tensors = tensors();
        tensors.get_mut("model.layers.1.mlp.gate_proj.weight").unwrap().1[0] = poison;
        let fixture = Fixture::new(&tensors, &config());
        let mut graph = fixture.load(&config()).unwrap();
        assert!(graph.step(0).is_err());
        assert_eq!(graph.position, 0);
        assert!(graph.graph.is_none());
    }
}

#[cfg(feature = "metal")]
#[stark_test]
fn unadmitted_dense_inventory_uses_staged_forward_and_resets_state() {
    let tensors = tensors();
    let fixture = Fixture::new(&tensors, &config());
    let expected = reference(&tensors, &[0, 1, 2, 3]);
    let mut model = fixture.load(&config()).unwrap();
    assert!(model.graph_admitted);
    assert!(model.graph_enabled);
    // Model a declined complete-view inventory without constructing a huge
    // fixture. The test selector remains enabled and cannot bypass admission.
    model.graph_admitted = false;
    assert!(!model.use_graph());
    model.ingest(0).unwrap();
    model.ingest(1).unwrap();
    close(&model.step(2).unwrap(), &expected[2]);
    assert!(model.graph.is_none());
    assert_eq!(model.states[0].keys.len(), 2);
    assert_eq!(model.states[1].keys.len(), 3);
    assert!(model.step(u32::MAX).is_err());
    assert_eq!(model.position, 0);
    assert!(model.graph.is_none());
    assert!(model.states.iter().all(|state| state.keys.is_empty() && state.values.is_empty()));
    close(&model.step(0).unwrap(), &expected[0]);
    model.reset();
    assert_eq!(model.position, 0);
    assert!(model.states.iter().all(|state| state.keys.is_empty() && state.values.is_empty()));
    close(&model.step(0).unwrap(), &expected[0]);
    assert!(!model.graph_admitted);
    assert!(model.graph.is_none());
}

#[cfg(feature = "metal")]
#[stark_test]
#[ignore]
fn full_model_control_modes_interleaved_permutations() {
    run_interleaved_control_modes([0, 1, 2], false);
}

#[cfg(feature = "metal")]
#[stark_test]
#[ignore]
fn full_model_dense_residency_interleaved_permutations() {
    run_interleaved_control_modes([0, 1, 3], true);
}

#[cfg(feature = "metal")]
fn run_interleaved_control_modes(
    selected_modes: [u8; 3],
    residency_experiment: bool,
) {
    use std::time::Instant;

    use sha2::{Digest, Sha256};

    use crate::engine::hybrid::{activity::InferenceActivity, profile};

    let root = PathBuf::from(std::env::var_os("STARK_RIPPLE_CONTROL_FIXTURE").expect("control_fixture_required"));
    let bounded_read = |name: &str, limit: u64| {
        let path = root.join(name);
        let meta = fs::symlink_metadata(&path).unwrap_or_else(|_| panic!("control_fixture_metadata"));
        assert!(meta.is_file() && !meta.file_type().is_symlink() && meta.len() <= limit, "control_fixture_bounds");
        fs::read(path).unwrap_or_else(|_| panic!("control_fixture_read"))
    };
    let config: Value = serde_json::from_slice(&bounded_read("config.json", 1024 * 1024))
        .unwrap_or_else(|_| panic!("control_fixture_config"));
    let tokens: Vec<u32> = serde_json::from_slice(&bounded_read("tokens.json", 16384))
        .unwrap_or_else(|_| panic!("control_fixture_tokens"));
    assert!(!tokens.is_empty() && tokens.len() <= 128, "control_fixture_token_bounds");
    let input = tokens.into_iter().cycle().take(8).collect::<Vec<_>>();
    let weights = Weights::open(&root).unwrap_or_else(|_| panic!("control_fixture_weights"));
    let mut model = RippleModel::new(weights, &config).unwrap_or_else(|_| panic!("control_fixture_model"));
    assert!(
        model.graph_admitted && model.graph_enabled && model.config.context >= 65,
        "control_fixture_graph_required"
    );
    assert!(input.iter().all(|&token| (token as usize) < model.config.vocab), "control_fixture_vocabulary");
    assert!(std::env::var_os("STARK_QWEN38_ENCODER_TIMING").is_none(), "control_timing_diagnostic_must_be_disabled");
    let _activity = InferenceActivity::begin().unwrap_or_else(|_| panic!("control_activity_unavailable"));
    let mut input_hash = Sha256::new();
    for &token in &input {
        input_hash.update(token.to_le_bytes());
    }
    let input_sha256 = format!("{:x}", input_hash.finalize());
    let mut projection_names = vec![model.output.clone()];
    for layer in &model.layers {
        projection_names.extend([
            layer.q.clone(),
            layer.k.clone(),
            layer.v.clone(),
            layer.o.clone(),
            layer.ffn_gate.clone(),
            layer.ffn_up.clone(),
            layer.ffn_down.clone(),
        ]);
        if let Some(gate) = &layer.gate {
            projection_names.push(gate.clone());
        }
    }
    let mut output_reference = None;
    let mut rates: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    let mut gpu_rates: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    let mut paired_ratios: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    let permutations = [[0u8, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    // A whole three-mode warmup precedes the six balanced measured rounds.
    // The model instance and its immutable weight mappings remain identical.
    for round in 0..=permutations.len() {
        let warmup = round == 0;
        let order = if warmup {
            selected_modes
        } else {
            permutations[round - 1].map(|index| selected_modes[usize::from(index)])
        };
        let mut round_rates = [0.0f64; 4];
        for (position, mode) in order.into_iter().enumerate() {
            model.test_control_mode = mode;
            model.reset();
            assert_eq!(model.test_control_mode, mode, "control_mode_reset_changed");
            profile::enable(false);
            let resident_bytes = if residency_experiment {
                model
                    .weights
                    .test_set_dense_residency(&projection_names, mode == 3)
                    .unwrap_or_else(|_| panic!("dense_residency_admission_failed"))
            } else {
                0
            };
            let prefill_started = Instant::now();
            let mut logits = Vec::new();
            for (index, &token) in input.iter().enumerate() {
                if index + 1 == input.len() {
                    logits = model.step(token).unwrap_or_else(|_| panic!("control_prefill_failed"));
                } else {
                    model.ingest(token).unwrap_or_else(|_| panic!("control_prefill_failed"));
                }
            }
            let prefill_seconds = prefill_started.elapsed().as_secs_f64();
            profile::reset();
            profile::enable(true);
            let started = Instant::now();
            let mut output_hash = Sha256::new();
            for _ in 0..57 {
                let mut best = 0;
                for index in 1..logits.len() {
                    if logits[index] > logits[best] {
                        best = index;
                    }
                }
                output_hash.update((best as u32).to_le_bytes());
                logits = model.step(best as u32).unwrap_or_else(|_| panic!("control_decode_failed"));
            }
            let decode_seconds = started.elapsed().as_secs_f64();
            profile::enable(false);
            let counters = profile::snapshot();
            let gpu = counters.get("native.metal.gpu").cloned().unwrap_or_default();
            let output_sha256 = format!("{:x}", output_hash.finalize());
            if !warmup {
                let rate = 57.0 / decode_seconds;
                rates[usize::from(mode)].push(rate);
                gpu_rates[usize::from(mode)].push(57.0 / gpu.seconds);
                round_rates[usize::from(mode)] = rate;
            }
            if let Some(reference) = &output_reference {
                assert!(reference == &output_sha256, "control_mode_generation_changed");
            } else {
                output_reference = Some(output_sha256.clone());
            }
            println!(
                "{}",
                json!({"scope":"single_model_interleaved_control_modes", "round":round,
                    "round_position":position,"mode":mode,"warmup":warmup,"residency_experiment":residency_experiment,"resident_static_bytes":resident_bytes,
                    "measured_rounds":6,"measured_requests_per_mode":6,
                    "input_tokens":8,"input_sha256":input_sha256,"decode_steps":57,
                    "prefill_seconds":prefill_seconds,"decode_seconds":decode_seconds,
                    "decode_tokens_per_second":57.0/decode_seconds,"gpu_seconds":gpu.seconds,
                    "gpu_completions":gpu.calls,"output_sha256":output_sha256,"passed":true})
            );
            model.reset();
        }
        if !warmup {
            for index in 0..2 {
                paired_ratios[index].push(
                    round_rates[usize::from(selected_modes[index + 1])] / round_rates[usize::from(selected_modes[0])],
                );
            }
            paired_ratios[2]
                .push(round_rates[usize::from(selected_modes[2])] / round_rates[usize::from(selected_modes[1])]);
        }
    }
    profile::enable(false);
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    };
    if residency_experiment {
        model
            .weights
            .test_set_dense_residency(&[], false)
            .unwrap_or_else(|_| panic!("dense_residency_teardown_failed"));
    }
    let summaries = selected_modes
        .into_iter()
        .map(|selected| {
            let mode = usize::from(selected);
            let rate = median(&mut rates[mode]);
            let gpu_rate = median(&mut gpu_rates[mode]);
            json!({"mode":mode,"measured_requests":rates[mode].len(),
                "median_decode_tokens_per_second":rate,"decode_rate_min":rates[mode][0],
                "decode_rate_max":rates[mode][rates[mode].len()-1],
                "median_gpu_tokens_per_second":gpu_rate})
        })
        .collect::<Vec<_>>();
    let ratios = (0..3)
        .map(|index| {
            let (mode, baseline_mode) = if index < 2 {
                (selected_modes[index + 1], selected_modes[0])
            } else {
                (selected_modes[2], selected_modes[1])
            };
            let ratio = median(&mut paired_ratios[index]);
            json!({"mode":mode,"baseline_mode":baseline_mode,"paired_rounds":6,
                "median_paired_decode_rate_ratio":ratio,
                "paired_ratio_min":paired_ratios[index][0],
                "paired_ratio_max":paired_ratios[index][5]})
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        json!({"scope":"single_model_interleaved_control_summary", "modes":summaries,"selected_modes":selected_modes,"residency_experiment":residency_experiment,
            "paired_round_ratios":ratios,"output_sha256":output_reference,
            "input_sha256":input_sha256,"passed":true})
    );
}

#[cfg(feature = "metal")]
#[stark_test]
fn diagnostic_snapshot_restores_exact_state_and_rejects_malformed_or_foreign() {
    let fixture = Fixture::new(&tensors(), &config());
    let mut model = fixture.load(&config()).unwrap();
    model.test_control_mode = 0;
    model.ingest(0).unwrap();
    model.ingest(1).unwrap();
    let mut snapshot = model.graph.as_ref().unwrap().test_snapshot(&model.config, model.position).unwrap();
    assert!(snapshot.payload_bytes > 0 && snapshot.payload_bytes <= 1024 * 1024 * 1024);
    let expected = model.step(2).unwrap();
    model.step(3).unwrap();
    let capacity = snapshot.layers[0].capacity;
    snapshot.layers[0].capacity = usize::MAX;
    assert!(model.graph.as_mut().unwrap().test_restore(&model.config, &snapshot).is_err());
    snapshot.layers[0].capacity = capacity;
    let original = snapshot.layers[0].keys[0];
    snapshot.layers[0].keys[0] = f32::NAN;
    assert!(model.graph.as_mut().unwrap().test_restore(&model.config, &snapshot).is_err());
    snapshot.layers[0].keys[0] = original;
    let mut other = fixture.load(&config()).unwrap();
    other.ingest(0).unwrap();
    assert!(other.graph.as_mut().unwrap().test_restore(&other.config, &snapshot).is_err());
    model.position = model.graph.as_mut().unwrap().test_restore(&model.config, &snapshot).unwrap();
    assert_eq!(model.position, 2);
    let actual = model.step(2).unwrap();
    assert!(actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[cfg(feature = "metal")]
#[stark_test]
#[ignore]
fn full_model_long_context_control_snapshot_comparison() {
    use std::time::Instant;

    use sha2::{Digest, Sha256};

    use crate::engine::hybrid::{activity::InferenceActivity, profile};

    const PREFIX: usize = 1024;
    const STEPS: usize = 57;
    let root = PathBuf::from(std::env::var_os("STARK_RIPPLE_CONTROL_FIXTURE").expect("control_fixture_required"));
    let read = |name: &str, limit: u64| {
        let path = root.join(name);
        let meta = fs::symlink_metadata(&path).unwrap_or_else(|_| panic!("long_control_fixture_metadata"));
        assert!(meta.is_file() && !meta.file_type().is_symlink() && meta.len() <= limit, "long_control_fixture_bounds");
        fs::read(path).unwrap_or_else(|_| panic!("long_control_fixture_read"))
    };
    let config: Value =
        serde_json::from_slice(&read("config.json", 1024 * 1024)).unwrap_or_else(|_| panic!("long_control_config"));
    let tokens: Vec<u32> =
        serde_json::from_slice(&read("tokens.json", 16384)).unwrap_or_else(|_| panic!("long_control_tokens"));
    assert!(!tokens.is_empty() && tokens.len() <= 128, "long_control_token_bounds");
    let input = tokens.into_iter().cycle().take(PREFIX).collect::<Vec<_>>();
    let mut model = RippleModel::new(Weights::open(&root).unwrap_or_else(|_| panic!("long_control_weights")), &config)
        .unwrap_or_else(|_| panic!("long_control_model"));
    assert!(
        model.graph_admitted && model.graph_enabled && model.config.context >= PREFIX + STEPS,
        "long_control_graph_required"
    );
    assert!(input.iter().all(|&token| (token as usize) < model.config.vocab), "long_control_vocabulary");
    assert!(
        model.config.vocab.checked_mul(STEPS).and_then(|n| n.checked_mul(4)).is_some_and(|n| n <= 64 * 1024 * 1024),
        "long_control_readout_cap"
    );
    assert!(std::env::var_os("STARK_QWEN38_ENCODER_TIMING").is_none(), "long_control_disable_timing_diagnostic");
    let _activity = InferenceActivity::begin().unwrap_or_else(|_| panic!("long_control_activity"));
    model.test_control_mode = 0;
    profile::enable(false);
    let prefill_started = Instant::now();
    let mut initial_logits = Vec::new();
    for (index, &token) in input.iter().enumerate() {
        if index + 1 == input.len() {
            initial_logits = model.step(token).unwrap_or_else(|_| panic!("long_control_prefill"));
        } else {
            model.ingest(token).unwrap_or_else(|_| panic!("long_control_prefill"));
        }
    }
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    let snapshot = model
        .graph
        .as_ref()
        .unwrap()
        .test_snapshot(&model.config, model.position)
        .unwrap_or_else(|_| panic!("long_control_snapshot"));
    assert_eq!(snapshot.position, PREFIX);
    let argmax = |values: &[f32]| {
        let mut best = 0;
        for index in 1..values.len() {
            if values[index] > values[best] {
                best = index;
            }
        }
        best
    };
    let readout_hash = |rows: &[Vec<f32>]| {
        let mut hash = Sha256::new();
        for row in rows {
            for value in row {
                hash.update(value.to_le_bytes());
            }
        }
        format!("{:x}", hash.finalize())
    };
    // The baseline warmup also captures a fixed teacher-forced continuation.
    // These token IDs/readout vectors remain private RAM and are never printed.
    let mut forced = Vec::with_capacity(STEPS);
    let mut reference: Vec<Vec<f32>> = Vec::with_capacity(STEPS);
    let reference_started = Instant::now();
    for index in 0..STEPS {
        let previous = if index == 0 {
            &initial_logits
        } else {
            &reference[index - 1]
        };
        let token = argmax(previous) as u32;
        forced.push(token);
        reference.push(model.step(token).unwrap_or_else(|_| panic!("long_control_reference")));
    }
    let reference_seconds = reference_started.elapsed().as_secs_f64();
    let reference_sha256 = readout_hash(&reference);
    let mut forced_hash = Sha256::new();
    for &token in &forced {
        forced_hash.update(token.to_le_bytes());
    }
    let forced_sha256 = format!("{:x}", forced_hash.finalize());
    println!(
        "{}",
        json!({"scope":"long_context_private_snapshot_reference","prefix_tokens":PREFIX,
        "decode_steps":STEPS,"prefill_seconds":prefill_seconds,"baseline_warmup_seconds":reference_seconds,
        "snapshot_payload_bytes":snapshot.payload_bytes,"snapshot_cap_bytes":1024*1024*1024,
        "forced_input_sha256":forced_sha256,"reference_sha256":reference_sha256})
    );
    let mut rates: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    let mut paired_ratios = Vec::new();
    // One candidate warmup, then four balanced ABBA/BAAB rounds. The captured
    // reference above is the single excluded baseline warmup.
    let orders = [[0u8, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]];
    for trial in 0..=16 {
        let warmup = trial == 0;
        let round = if warmup {
            0
        } else {
            (trial - 1) / 4 + 1
        };
        let round_position = if warmup {
            0
        } else {
            (trial - 1) % 4
        };
        let mode = if warmup {
            1
        } else {
            orders[round - 1][round_position]
        };
        model.test_control_mode = mode;
        profile::enable(false);
        model.position = model
            .graph
            .as_mut()
            .unwrap()
            .test_restore(&model.config, &snapshot)
            .unwrap_or_else(|_| panic!("long_control_restore"));
        assert_eq!(model.position, PREFIX);
        let mut readouts = Vec::with_capacity(STEPS);
        profile::reset();
        profile::enable(true);
        let started = Instant::now();
        for &token in &forced {
            readouts.push(model.step(token).unwrap_or_else(|_| panic!("long_control_decode")));
        }
        let seconds = started.elapsed().as_secs_f64();
        profile::enable(false);
        let counters = profile::snapshot();
        let gpu = counters.get("native.metal.gpu").cloned().unwrap_or_default();
        let mut max_abs = 0f64;
        let mut sum_abs = 0f64;
        let mut squared = 0f64;
        let mut reference_squared = 0f64;
        let mut greedy_agreement = 0usize;
        let mut within_tolerance = true;
        for (actual, expected) in readouts.iter().zip(&reference) {
            greedy_agreement += usize::from(argmax(actual) == argmax(expected));
            for (&a, &b) in actual.iter().zip(expected) {
                let error = (f64::from(a) - f64::from(b)).abs();
                max_abs = max_abs.max(error);
                sum_abs += error;
                squared += error * error;
                reference_squared += f64::from(b).powi(2);
                within_tolerance &= a.is_finite() && error <= 0.001 + 0.001 * f64::from(b).abs();
            }
        }
        assert!(within_tolerance, "long_control_readout_tolerance");
        let output_sha256 = readout_hash(&readouts);
        if mode == 0 {
            assert!(output_sha256 == reference_sha256, "long_control_restore_not_exact");
        }
        let rate = STEPS as f64 / seconds;
        if !warmup {
            rates[usize::from(mode)].push(rate);
        }
        println!(
            "{}",
            json!({"scope":"long_context_snapshot_decode","warmup":warmup,"round":round,
            "round_position":round_position,"mode":mode,"prefix_tokens":PREFIX,"decode_steps":STEPS,
            "decode_seconds":seconds,"decode_tokens_per_second":rate,"gpu_seconds":gpu.seconds,"gpu_completions":gpu.calls,
            "readouts":STEPS,"vocabulary":model.config.vocab,"max_abs_error":max_abs,
            "mean_abs_error":sum_abs/(STEPS*model.config.vocab) as f64,"relative_l2_error":(squared/reference_squared).sqrt(),
            "greedy_agreement":greedy_agreement,"reference_sha256":reference_sha256,"readout_sha256":output_sha256,
            "forced_input_sha256":forced_sha256,"passed":true})
        );
        if !warmup && round_position == 3 {
            let a = &rates[0][rates[0].len() - 2..];
            let b = &rates[1][rates[1].len() - 2..];
            paired_ratios.push((b[0] + b[1]) / (a[0] + a[1]));
        }
    }
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    };
    let a = median(&mut rates[0]);
    let b = median(&mut rates[1]);
    let ratio = median(&mut paired_ratios);
    println!(
        "{}",
        json!({"scope":"long_context_snapshot_summary","prefix_tokens":PREFIX,"decode_steps":STEPS,
        "measured_requests_per_mode":8,"warmups_per_mode":1,"baseline_median_tokens_per_second":a,
        "candidate_median_tokens_per_second":b,"median_paired_round_ratio":ratio,
        "paired_ratio_min":paired_ratios[0],"paired_ratio_max":paired_ratios[paired_ratios.len()-1],
        "snapshot_payload_bytes":snapshot.payload_bytes,"forced_input_sha256":forced_sha256,"passed":true})
    );
    profile::enable(false);
}
