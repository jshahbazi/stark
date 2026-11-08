use std::{collections::BTreeMap, fs, path::PathBuf};

use serde_json::json;
use stark_engine_macros::stark_test;

use super::*;

type Tensors = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

fn config() -> Value {
    json!({"model_type":"flint", "hidden_act":"silu", "attention_bias":false,
        "attention_dropout":0.0, "tie_word_embeddings":false, "norm_topk_prob":true,
        "softmax_type":"learnable", "decoder_sparse_step":1, "use_sliding_window":true,
        "router_score_function":"sigmoid", "hidden_size":4, "num_attention_heads":2,
        "num_key_value_heads":1, "head_dim":2, "num_hidden_layers":3,
        "layer_types":["full_attention","sliding_attention","full_attention"],
        "no_rope_layer_indices":[0,2], "mlp_only_layers":[0], "intermediate_size":3,
        "moe_intermediate_size":3, "shared_expert_intermediate_size":2,
        "num_experts":3, "num_experts_per_tok":2, "vocab_size":5,
        "max_position_embeddings":6, "sliding_window":2, "rms_norm_eps":1e-6,
        "rope_theta":10000.0})
}

fn tensors() -> Tensors {
    let c = Config::parse(&config()).unwrap();
    let mut tensors = Tensors::new();
    let mut add = |name: String, shape: Vec<usize>| {
        let count = shape.iter().product();
        let seed = name.bytes().map(usize::from).sum::<usize>();
        let values = (0..count)
            .map(|i| {
                let v = ((i * 17 + seed * 7) % 53) as f32 - 26.0;
                if name.ends_with("norm.weight") || name.contains("layernorm.weight") {
                    1.0 + v * 0.008
                } else if name.ends_with("softmax_offset") {
                    v * 0.025
                } else if name.ends_with("expert_bias") {
                    v * 0.08
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
    for i in 0..c.layers.len() {
        let p = format!("model.layers.{i}");
        for name in ["input_layernorm", "post_attention_layernorm"] {
            add(format!("{p}.{name}.weight"), vec![c.hidden]);
        }
        for name in ["q", "k"] {
            add(format!("{p}.self_attn.{name}_norm.weight"), vec![c.dim]);
        }
        add(format!("{p}.self_attn.softmax_offset"), vec![c.heads]);
        add(format!("{p}.self_attn.q_proj.weight"), vec![c.heads * c.dim, c.hidden]);
        for name in ["k", "v"] {
            add(format!("{p}.self_attn.{name}_proj.weight"), vec![c.kv_heads * c.dim, c.hidden]);
        }
        add(format!("{p}.self_attn.o_proj.weight"), vec![c.hidden, c.heads * c.dim]);
        let mut dense = |prefix: String, intermediate| {
            for name in ["gate", "up"] {
                add(format!("{prefix}.{name}_proj.weight"), vec![intermediate, c.hidden]);
            }
            add(format!("{prefix}.down_proj.weight"), vec![c.hidden, intermediate]);
        };
        if c.layers[i].dense {
            dense(format!("{p}.mlp"), c.intermediate);
        } else {
            dense(format!("{p}.mlp.shared_expert"), c.shared_intermediate);
            for expert in 0..c.experts {
                dense(format!("{p}.mlp.experts.{expert}"), c.expert_intermediate);
            }
            add(format!("{p}.mlp.gate.weight"), vec![c.experts, c.hidden]);
            add(format!("{p}.mlp.gate.expert_bias"), vec![c.experts]);
        }
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
            "stark-flint-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        let mut metadata = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, (shape, values)) in tensors {
            let start = payload.len();
            payload.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            metadata.insert(name.clone(), json!({"dtype":"F32", "shape":shape, "data_offsets":[start,payload.len()]}));
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
    ) -> Result<FlintModel> {
        FlintModel::new(Weights::open(&self.0)?, config)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

// Independent f64 reference: dense source tensors and full causal histories,
// without production Weights, RMS/softmax/routing helpers, or rotating queues.
fn reference(
    tensors: &Tensors,
    sequence: &[u32],
) -> Vec<Vec<f64>> {
    let t = |name: &str| tensors[name].1.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let matrix = |name: &str, x: &[f64]| {
        let (shape, values) = &tensors[name];
        values
            .chunks_exact(shape[1])
            .map(|row| row.iter().zip(x).map(|(&w, &x)| f64::from(w) * x).sum::<f64>())
            .collect::<Vec<_>>()
    };
    let norm = |name: &str, x: &[f64]| {
        let scale = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + f64::from(1e-6f32)).sqrt().recip();
        x.iter().zip(t(name)).map(|(&v, w)| v * scale * w).collect::<Vec<_>>()
    };
    let mlp = |prefix: &str, x: &[f64]| {
        let g = matrix(&format!("{prefix}.gate_proj.weight"), x);
        let u = matrix(&format!("{prefix}.up_proj.weight"), x);
        let h: Vec<_> = g.iter().zip(u).map(|(&g, u)| g / (1.0 + (-g).exp()) * u).collect();
        matrix(&format!("{prefix}.down_proj.weight"), &h)
    };
    let mut keys: Vec<Vec<Vec<f64>>> = vec![Vec::new(); 3];
    let mut values = keys.clone();
    let mut outputs = Vec::new();
    for (position, &token) in sequence.iter().enumerate() {
        let mut x = t("model.embed_tokens.weight")[token as usize * 4..token as usize * 4 + 4].to_vec();
        for layer in 0..3 {
            let p = format!("model.layers.{layer}");
            let n = norm(&format!("{p}.input_layernorm.weight"), &x);
            let a = format!("{p}.self_attn");
            let mut q = matrix(&format!("{a}.q_proj.weight"), &n);
            let mut k = matrix(&format!("{a}.k_proj.weight"), &n);
            let v = matrix(&format!("{a}.v_proj.weight"), &n);
            for (heads, name) in [(&mut q, "q"), (&mut k, "k")] {
                for head in heads.chunks_exact_mut(2) {
                    let mut normalized = norm(&format!("{a}.{name}_norm.weight"), head);
                    if layer == 1 {
                        let (s, c) = (position as f64).sin_cos();
                        normalized = vec![normalized[0] * c - normalized[1] * s, normalized[0] * s + normalized[1] * c];
                    }
                    head.copy_from_slice(&normalized);
                }
            }
            keys[layer].push(k);
            values[layer].push(v);
            let start = if layer == 1 {
                (position + 1).saturating_sub(2)
            } else {
                0
            };
            let sinks = t(&format!("{a}.softmax_offset"));
            let mut attn = vec![0.0; 4];
            for h in 0..2 {
                let mut scores: Vec<_> = keys[layer][start..]
                    .iter()
                    .map(|k| (q[h * 2] * k[0] + q[h * 2 + 1] * k[1]) / 2f64.sqrt())
                    .collect();
                scores.push(sinks[h]);
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<_> = scores.iter().map(|v| (v - max).exp()).collect();
                let sum = exps.iter().sum::<f64>();
                for (weight, row) in exps.iter().zip(&values[layer][start..]) {
                    for j in 0..2 {
                        attn[h * 2 + j] += weight / sum * row[j];
                    }
                }
            }
            let attn = matrix(&format!("{a}.o_proj.weight"), &attn);
            for j in 0..4 {
                x[j] += attn[j];
            }
            let n = norm(&format!("{p}.post_attention_layernorm.weight"), &x);
            let m = format!("{p}.mlp");
            let update = if layer == 0 {
                mlp(&m, &n)
            } else {
                let logits = matrix(&format!("{m}.gate.weight"), &n);
                let scores: Vec<_> = logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
                let bias = t(&format!("{m}.gate.expert_bias"));
                let mut order: Vec<_> = (0..3).collect();
                order.sort_by(|&a, &b| (scores[b] + bias[b]).total_cmp(&(scores[a] + bias[a])).then_with(|| a.cmp(&b)));
                order.truncate(2);
                let sum = order.iter().map(|&i| scores[i]).sum::<f64>() + 1e-20;
                let mut result = mlp(&format!("{m}.shared_expert"), &n);
                for expert in order {
                    let expert_values = mlp(&format!("{m}.experts.{expert}"), &n);
                    for j in 0..4 {
                        result[j] += expert_values[j] * scores[expert] / sum;
                    }
                }
                result
            };
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
    assert!(
        actual.iter().zip(expected).all(|(&a, &b)| (f64::from(a) - b).abs() < 5e-6),
        "synthetic Flint numerical mismatch"
    );
}

fn as_banks(mut tensors: Tensors) -> Tensors {
    for layer in [1, 2] {
        for kind in ["gate", "up", "down"] {
            let p = format!("model.layers.{layer}.mlp");
            let mut values = Vec::new();
            let mut shape = Vec::new();
            for expert in 0..3 {
                let (s, v) = tensors.remove(&format!("{p}.experts.{expert}.{kind}_proj.weight")).unwrap();
                shape = s;
                values.extend(v);
            }
            shape.insert(0, 3);
            tensors.insert(format!("{p}.switch_mlp.{kind}_proj.weight"), (shape, values));
        }
    }
    tensors
}

#[stark_test]
fn flint_multistep_matches_independent_dense_reference_and_ingest() {
    let c = config();
    let source = tensors();
    let sequence = [0, 1, 3, 2, 4, 1];
    let expected = reference(&source, &sequence);
    for data in [source.clone(), as_banks(source)] {
        let fixture = Fixture::new(&data, &c);
        let mut model = fixture.load(&c).unwrap();
        for (&token, expected) in sequence.iter().zip(&expected) {
            close(&model.step(token).unwrap(), expected);
        }
        assert_eq!(model.states[0].keys.len(), 6);
        assert_eq!(model.states[1].keys.len(), 2);
        assert_eq!(model.states[2].keys.len(), 6);
        assert!(model.step(0).is_err());
        assert_eq!(model.position, 0);
        assert!(model.states.iter().all(|state| state.keys.is_empty() && state.values.is_empty()));
        close(&model.step(sequence[0]).unwrap(), &expected[0]);
        assert!(model.ingest(u32::MAX).is_err());
        for end in 0..sequence.len() {
            model.reset();
            for &token in &sequence[..end] {
                model.ingest(token).unwrap();
            }
            close(&model.step(sequence[end]).unwrap(), &expected[end]);
        }
    }
}

#[stark_test]
fn flint_rope_uses_split_halves_and_preserves_no_rope_layers() {
    let mut values = [1.0, 2.0, 3.0, 4.0];
    rope(&mut values, 1, 10000.0);
    let expected = [
        1.0 * 1f64.cos() - 3.0 * 1f64.sin(),
        2.0 * 0.01f64.cos() - 4.0 * 0.01f64.sin(),
        1.0 * 1f64.sin() + 3.0 * 1f64.cos(),
        2.0 * 0.01f64.sin() + 4.0 * 0.01f64.cos(),
    ];
    close(&values, &expected);
    let parsed = Config::parse(&config()).unwrap();
    assert!(!parsed.layers[0].rope && parsed.layers[1].rope && !parsed.layers[2].rope);
}

#[stark_test]
fn flint_routing_bias_changes_choice_only_and_zero_scores_stay_finite() {
    let (selected, weights) = route(&[0.0, 1.0, 2.0], &[1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(selected, [0, 2]);
    let denominator = 0.5 + 1.0 / (1.0 + (-2f64).exp());
    close(&weights, &[0.5 / denominator, (denominator - 0.5) / denominator]);
    let (_, weights) = route(&[-1000.0, -1000.0], &[0.0, 0.0], 1).unwrap();
    assert!(weights.iter().all(|v| v.is_finite()));
    assert!(route(&[f32::NAN], &[0.0], 1).is_err());
    assert!(route(&[0.0], &[f32::INFINITY], 1).is_err());
}

#[stark_test]
fn flint_rejects_unsupported_operations_and_invalid_weights() {
    let baseline = config();
    for (field, value) in [
        ("norm_topk_prob", json!(false)),
        ("attention_bias", json!(true)),
        ("softmax_type", json!("vanilla")),
        ("router_score_function", json!("softmax")),
        ("num_key_value_heads", json!(3)),
        ("head_dim", json!(3)),
        ("layer_types", json!(["full_attention"])),
        ("mlp_only_layers", json!([0, 0])),
        ("no_rope_layer_indices", json!([9])),
        ("rope_scaling", json!({"type":"linear","factor":2})),
        ("rope_parameters", json!({"rope_type":"yarn","rope_theta":10000.0})),
        ("scoring_func", json!("softmax")),
    ] {
        let mut changed = baseline.clone();
        changed[field] = value;
        assert!(Config::parse(&changed).is_err());
    }
    let mut data = tensors();
    data.get_mut("model.layers.0.self_attn.softmax_offset").unwrap().0 = vec![1, 2];
    let fixture = Fixture::new(&data, &baseline);
    assert!(fixture.load(&baseline).is_err());
    let mut data = tensors();
    data.get_mut("model.layers.1.mlp.gate.expert_bias").unwrap().1[0] = f32::NAN;
    let fixture = Fixture::new(&data, &baseline);
    let mut model = fixture.load(&baseline).unwrap();
    assert!(model.step(0).is_err());
    assert_eq!(model.position, 0);
    assert!(model.states.iter().all(|state| state.keys.is_empty()));
}

fn prefix_config() -> Value {
    let mut c = config();
    c["max_position_embeddings"] = json!(4096);
    c
}

fn prefix_rows(
    config: &Config,
    position: usize,
) -> Vec<AttentionState> {
    let window = config.window;
    config
        .layers
        .iter()
        .enumerate()
        .map(|(layer, config)| {
            let start = if config.sliding {
                position.saturating_sub(window)
            } else {
                0
            };
            let keys = (start..position)
                .map(|row| vec![((row + layer) % 17) as f32 * 0.01, (row % 11) as f32 * -0.02])
                .collect();
            let values = (start..position)
                .map(|row| vec![(row % 7) as f32 * -0.03, ((row + layer) % 13) as f32 * 0.02])
                .collect();
            AttentionState {
                keys,
                values,
            }
        })
        .collect()
}

fn same_attention_state(
    actual: &[AttentionState],
    expected: &[AttentionState],
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        for (actual, expected) in [(&actual.keys, &expected.keys), (&actual.values, &expected.values)] {
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| { a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()) }),
                "Flint private attention state differs"
            );
        }
    }
}

#[stark_test]
fn flint_prefix_snapshot_copies_full_and_rotating_states_at_complete_token() {
    let c = prefix_config();
    let fixture = Fixture::new(&tensors(), &c);
    let mut model = fixture.load(&c).unwrap();
    let chunk = model.prefill_chunk_tokens();
    model.capture_prefix_at(chunk).unwrap();
    model.position = chunk - 1;
    model.states = prefix_rows(&model.config, model.position);
    assert!(model.take_prefix_snapshot().is_none());
    model.ingest(3).unwrap();
    let saved = model.take_prefix_snapshot().expect("complete Flint prefix snapshot");
    assert_eq!(saved.position(), chunk);
    assert!(model.take_prefix_snapshot().is_none());
    let expected = model.states.clone();
    model.ingest(4).unwrap();
    model.ingest(1).unwrap();
    model.reset();
    model.restore_prefix(&saved).unwrap();
    assert_eq!(model.position, chunk);
    same_attention_state(&model.states, &expected);
    let actual = model.step(2).unwrap();
    let mut fresh = fixture.load(&c).unwrap();
    fresh.position = chunk;
    fresh.states = expected;
    let reference = fresh.step(2).unwrap();
    assert!(actual.iter().zip(reference).all(|(a, b)| a.to_bits() == b.to_bits()), "Flint restored readout differs");

    model.reset();
    model.restore_prefix(&saved).unwrap();
    model.capture_prefix_at(chunk * 2).unwrap();
    model.position = chunk * 2;
    model.states = prefix_rows(&model.config, model.position);
    model.capture_prefix_if_ready().unwrap();
    let later = model.take_prefix_snapshot().unwrap();
    assert_eq!(later.position(), chunk * 2);
    model.reset();
    model.restore_prefix(&saved).unwrap();
    same_attention_state(&model.states, &saved.states);
    model.reset();
    model.restore_prefix(&later).unwrap();
    same_attention_state(&model.states, &later.states);
}

#[stark_test]
fn flint_prefix_rejects_foreign_and_malformed_snapshots_and_clears_state() {
    let c = prefix_config();
    let fixture = Fixture::new(&tensors(), &c);
    let mut model = fixture.load(&c).unwrap();
    let chunk = model.prefill_chunk_tokens();
    model.capture_prefix_at(chunk).unwrap();
    model.position = chunk;
    model.states = prefix_rows(&model.config, chunk);
    model.capture_prefix_if_ready().unwrap();
    let saved = model.take_prefix_snapshot().unwrap();
    let mut other = fixture.load(&c).unwrap();
    assert!(other.restore_prefix(&saved).is_err());
    assert_eq!(other.position, 0);
    assert!(other.states.iter().all(|state| state.keys.is_empty() && state.values.is_empty()));

    for changed in 0..10 {
        let mut invalid = PrefixSnapshot {
            instance: Arc::clone(&saved.instance),
            position: saved.position,
            states: saved.states.clone(),
        };
        match changed {
            0 => invalid.position = 0,
            1 => invalid.position += 1,
            2 => invalid.position = 4096,
            3 => {
                invalid.states.pop();
            },
            4 => {
                invalid.states[0].keys.pop_back();
            },
            5 => {
                invalid.states[1].values.push_back(vec![0.0; 2]);
            },
            6 => {
                invalid.states[0].keys[0].pop();
            },
            7 => invalid.states[1].values[0][0] = f32::NAN,
            8 => invalid.states[2].values[0][0] = f32::INFINITY,
            9 => invalid.instance = Arc::new(()),
            _ => unreachable!(),
        }
        model.reset();
        assert!(model.restore_prefix(&invalid).is_err());
        assert_eq!(model.position, 0);
        assert!(model.states.iter().all(|state| state.keys.is_empty() && state.values.is_empty()));
        model.restore_prefix(&saved).unwrap();
        assert!(model.restore_prefix(&saved).is_err(), "restore must start from reset state");
        assert_eq!(model.position, 0);
    }
    for position in [0, 1, chunk - 1, chunk + 1, 4096] {
        model.restore_prefix(&saved).unwrap();
        assert!(model.capture_prefix_at(position).is_err());
        assert_eq!(model.position, 0);
        assert!(model.take_prefix_snapshot().is_none());
    }
    model.capture_prefix_at(chunk).unwrap();
    model.reset();
    model.position = chunk;
    model.states = prefix_rows(&model.config, chunk);
    model.capture_prefix_if_ready().unwrap();
    assert!(model.take_prefix_snapshot().is_none(), "reset clears a pending capture");
    model.reset();
    model.capture_prefix_at(chunk).unwrap();
    assert!(model.ingest(u32::MAX).is_err());
    assert!(model.take_prefix_snapshot().is_none());
    assert!(model.capture_at.is_none());
}

// The full 1024-token scalar replay is model-free. Metal is qualified on the
// actual artifact separately; the state tests above cover both compiled paths.
#[cfg(not(feature = "metal"))]
#[stark_test]
fn flint_prefix_cpu_full_prefill_matches_fresh_bits_after_sliding_eviction() {
    let c = prefix_config();
    let fixture = Fixture::new(&as_banks(tensors()), &c);
    let mut model = fixture.load(&c).unwrap();
    let chunk = model.prefill_chunk_tokens();
    let input: Vec<_> = (0..chunk + 3).map(|i| (i % 5) as u32).collect();
    model.capture_prefix_at(chunk).unwrap();
    for &token in &input[..input.len() - 1] {
        model.ingest(token).unwrap();
    }
    let expected = model.step(input[input.len() - 1]).unwrap();
    let saved = model.take_prefix_snapshot().unwrap();
    assert_eq!(saved.position(), chunk);
    model.reset();
    model.restore_prefix(&saved).unwrap();
    for &token in &input[chunk..input.len() - 1] {
        model.ingest(token).unwrap();
    }
    let actual = model.step(input[input.len() - 1]).unwrap();
    assert!(actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()), "Flint full-prefix readout differs");
}
