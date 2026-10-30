//! Private synthetic artifact coverage. No production model data is used.

use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use half::bf16;
use stark_engine_macros::stark_test;

#[cfg(backend = "cpu")]
use super::super::Qwen38WeightLoader;
use super::*;
#[cfg(backend = "cpu")]
use crate::backends::{
    common::{BufferRef, Context},
    cpu::Cpu,
};
use crate::config::model::qwen38_language_model::Qwen38TensorBinding;
#[cfg(backend = "cpu")]
type FixtureBackend = Cpu;
#[cfg(all(not(backend = "cpu"), backend = "metal"))]
type FixtureBackend = crate::backends::metal::Metal;

fn config() -> Qwen38DecoderConfig {
    let parts = [
        serde_json::json!({"hidden_size":64,"num_hidden_layers":2,"layer_types":["linear_attention","full_attention"],
            "vocab_size":128,"rms_norm_eps":0.000001,"max_position_embeddings":64,"tie_word_embeddings":false,"eos_token_id":[2]}),
        serde_json::json!({"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":64,"shared_expert_intermediate_size":64,
            "norm_topk_prob":true,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "rope_parameters":{"rope_theta":10000.0,"partial_rotary_factor":1.0,"mrope_section":null,"mrope_interleaved":null,"rope_type":null}}),
        serde_json::json!({"linear_num_key_heads":1,"linear_num_value_heads":2,"linear_key_head_dim":32,"linear_value_head_dim":32,
            "linear_conv_kernel_dim":4,"output_gate_type":"silu","hidden_act":"silu","hc_count":4,"hc_lowrank":64}),
        serde_json::json!({"ple_layer_ids":[1],"ple_embed_dim":64,"ple_conv_kernel_size":4,"ngram_size":3,"heads_per_ngram":1,
            "split_ngram_parts":2,"indexer_n_heads":1,"indexer_kv_heads":1,"indexer_head_dim":64,"indexer_budget":8,"indexer_compress_ratio":4}),
        serde_json::json!({"ple":{"layer_multipliers":[1,2,18446744073709551615_u64],"ngram_heads_vocab_sizes":[3,5],"ngram_heads_offsets":[0,8],"padded_vocab_size":16},
            "quantization":{"default":{"bits":4,"group_size":64},"ngram_embedding":{"bits":4,"group_size":32},"q8_groups":["attn","gdn","shared","head","embed"]}}),
    ];
    let mut architecture = serde_json::Map::new();
    for part in parts {
        architecture.extend(part.as_object().unwrap().clone());
    }
    serde_json::from_value(serde_json::json!({"architecture":architecture,"tensor_bindings":{}})).unwrap()
}

pub(crate) struct Fixture {
    pub(crate) directory: PathBuf,
    pub(crate) path: PathBuf,
    pub(crate) cfg: Qwen38DecoderConfig,
    header: serde_json::Value,
    payload: Vec<u8>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}
impl Fixture {
    #[cfg(backend = "cpu")]
    fn new() -> Self {
        Self::from_config(config(), false)
    }

    pub(crate) fn complete() -> Self {
        let mut cfg = config();
        let a = &mut cfg.architecture;
        a.max_position_embeddings = 16;
        a.num_attention_heads = 12;
        a.num_key_value_heads = 1;
        a.head_dim = 256;
        a.rope_parameters.partial_rotary_factor = 0.25;
        a.linear_key_head_dim = 128;
        a.linear_value_head_dim = 128;
        a.indexer_n_heads = 4;
        a.indexer_head_dim = 128;
        a.indexer_budget = 2;
        a.indexer_compress_ratio = 2;
        a.ple_layer_ids = vec![2].into();
        a.ple_embed_dim = Some(512);
        a.heads_per_ngram = 8;
        a.ple.layer_multipliers = vec![1, 0x9e3779b97f4a7c15, 0xc2b2ae3d27d4eb4f].into();
        a.ple.ngram_heads_vocab_sizes = (0..16).map(|head| 13 + (head % 3) as u64).collect();
        a.ple.ngram_heads_offsets = (0..16).map(|head| head * 16).collect();
        a.ple.padded_vocab_size = 256;
        let fixture = Self::from_config(cfg, true);
        fixture.write_model_metadata();
        fixture
    }

    /// Isolate state/order semantics from BF16 GEMM weight staging: centered
    /// Q4/Q8 integers times a power of two are exactly BF16 representable.
    pub(crate) fn complete_bf16_exact_affines() -> Self {
        let mut fixture = Self::complete();
        fixture.cfg.architecture.max_position_embeddings = 48;
        fixture.cfg.architecture.num_experts_per_tok = 3;
        let inventory = inventory::<FixtureBackend>(&fixture.cfg).unwrap();
        for matrix in inventory.matrices.values() {
            let groups = matrix.columns / matrix.group_size;
            let padded_rows = matrix.rows.next_multiple_of(4);
            let scale_name = format!("{}.scales", matrix.physical);
            let bias_name = format!("{}.biases", matrix.physical);
            let scale_start = fixture.header[&scale_name]["data_offsets"][0].as_u64().unwrap() as usize;
            let bias_start = fixture.header[&bias_name]["data_offsets"][0].as_u64().unwrap() as usize;
            let salt = matrix.physical.bytes().fold(0_u32, |value, byte| value.wrapping_add(byte as u32));
            for row in 0..matrix.rows {
                for group in 0..groups {
                    let exponent = -12 + ((row + group * 3 + salt) % 5) as i32;
                    let scale = bf16::from_f32(2.0_f32.powi(exponent));
                    let bias = bf16::from_f32(-scale.to_f32() * (1_u32 << (matrix.bits - 1)) as f32);
                    let index = if matrix.embedding {
                        row * groups + group
                    } else {
                        group * padded_rows + row
                    } as usize;
                    fixture.payload[scale_start + index * 2..scale_start + index * 2 + 2]
                        .copy_from_slice(&scale.to_bits().to_le_bytes());
                    fixture.payload[bias_start + index * 2..bias_start + index * 2 + 2]
                        .copy_from_slice(&bias.to_bits().to_le_bytes());
                }
            }
        }
        fixture.write(None);
        fixture.write_model_metadata();
        fixture
    }

    fn from_config(
        mut cfg: Qwen38DecoderConfig,
        nonzero: bool,
    ) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "stark-qwen38-weight-fixture-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&directory).unwrap();
        let path = directory.join("model.safetensors");
        let initial = inventory::<FixtureBackend>(&cfg).unwrap();
        for (logical, plane) in initial.planes {
            cfg.tensor_bindings.0.insert(
                logical,
                Qwen38TensorBinding {
                    name: plane.physical,
                    dtype: plane.logical_dtype,
                    shape: plane.logical_shape.into(),
                    layout: plane.layout,
                },
            );
        }
        let f32_control = "model.language_model.layers.0.linear_attn.A_log";
        cfg.tensor_bindings.0.get_mut(f32_control).unwrap().dtype = Qwen38TensorDType::F32;
        let ple_layer = cfg.architecture.ple_layer_ids[0] - 1;
        for (name, width, dtype) in [
            ("layer_multipliers", cfg.architecture.ngram_size, Qwen38TensorDType::I64),
            (
                "ngram_heads_vocab_sizes",
                (cfg.architecture.ngram_size - 1) * cfg.architecture.heads_per_ngram,
                Qwen38TensorDType::U64,
            ),
            (
                "ngram_heads_offsets",
                (cfg.architecture.ngram_size - 1) * cfg.architecture.heads_per_ngram,
                Qwen38TensorDType::I64,
            ),
        ] {
            let logical = format!("model.language_model.layers.{ple_layer}.ple.ple_embedding.{name}");
            cfg.tensor_bindings.0.insert(
                logical.clone(),
                Qwen38TensorBinding {
                    name: canonical(&logical),
                    dtype,
                    shape: vec![width].into(),
                    layout: Qwen38TensorLayout::Direct,
                },
            );
        }
        let inventory = inventory::<FixtureBackend>(&cfg).unwrap();
        let mut metadata = serde_json::Map::new();
        metadata.insert("stark.qwen38.schema_version".into(), "1".into());
        metadata.insert(
            "stark.qwen38.omitted".into(),
            serde_json::json!({"vision":{"bytes":0,"tensors":0},"mtp":{"bytes":0,"tensors":0}}).to_string().into(),
        );
        for matrix in inventory.matrices.values() {
            metadata.insert(format!("{}.spec",matrix.physical),serde_json::json!({"type":"MLXSpec","bits":matrix.bits,"group_size":matrix.group_size,"layout":if matrix.embedding{"input_output"}else{"output_input"}}).to_string().into());
        }
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), metadata.into());
        let mut payload = Vec::new();
        for plane in inventory.planes.values() {
            let count = plane.physical_shape.iter().map(|&dim| dim as usize).product::<usize>();
            let start = payload.len();
            let mut bytes = vec![0_u8; count * plane.physical_dtype.size_in_bytes()];
            if plane.physical_dtype == DataType::U8 {
                bytes.fill(0x12);
            } else if plane.physical_dtype == DataType::BF16 {
                for word in bytes.chunks_exact_mut(2) {
                    word.copy_from_slice(&0x3f40_u16.to_le_bytes());
                }
            } else if plane.physical_dtype == DataType::F32 {
                for word in bytes.chunks_exact_mut(4) {
                    word.copy_from_slice(&0x80000000_u32.to_le_bytes());
                }
            }
            if nonzero {
                fill_nonzero(plane, &inventory.matrices, &mut bytes);
            }
            if let Some((rows, groups, stride)) = plane.padding {
                for group in 0..groups {
                    bytes[2 * (group * stride + rows) as usize..2 * ((group + 1) * stride) as usize].fill(0);
                }
            }
            if let Some(values) = &plane.hash {
                for (word, value) in bytes.chunks_exact_mut(8).zip(values) {
                    word.copy_from_slice(&value.to_le_bytes());
                }
            }
            payload.extend(bytes);
            header.insert(plane.physical.clone(),serde_json::json!({"dtype":dtype_name(plane.physical_dtype),"shape":plane.physical_shape,"data_offsets":[start,payload.len()]}));
        }
        let fixture = Self {
            directory,
            path,
            cfg,
            header: header.into(),
            payload,
        };
        fixture.write(None);
        fixture
    }
    fn write(
        &self,
        raw_header: Option<String>,
    ) {
        let mut encoded = raw_header.unwrap_or_else(|| self.header.to_string()).into_bytes();
        encoded.resize(encoded.len().next_multiple_of(8), b' ');
        let mut options = OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&self.path).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&encoded).unwrap();
        file.write_all(&self.payload).unwrap();
    }
    fn write_model_metadata(&self) {
        let generation = serde_json::json!({"stop_token_ids":[2],"temperature":null,"top_k":null,"top_p":null,
            "min_p":null,"banned_tokens":null,"repetition_penalty":null,"presence_penalty":null,
            "frequency_penalty":null,"suffix_repetition_length":null});
        let root = serde_json::json!({"type":"Qwen38LanguageModelConfig","token_codec_config":{"type":"RawTextCodecConfig"},
            "decoder_config":&self.cfg,"generation_config":generation});
        std::fs::write(self.directory.join("config.json"), serde_json::to_vec(&root).unwrap()).unwrap();
        let vocab = (0..self.cfg.architecture.vocab_size).map(|id| (format!("t{id}"), id)).collect();
        let tokenizer = tokenizers::Tokenizer::new(
            tokenizers::models::wordlevel::WordLevel::builder().vocab(vocab).build().unwrap(),
        );
        tokenizer.save(self.directory.join("tokenizer.json"), false).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for name in ["config.json", "tokenizer.json"] {
                std::fs::set_permissions(self.directory.join(name), std::fs::Permissions::from_mode(0o600)).unwrap();
            }
        }
    }
    pub(crate) fn zero_matrix(
        &mut self,
        physical: &str,
    ) {
        for suffix in ["scales", "biases"] {
            let offsets = self.header[format!("{physical}.{suffix}")]["data_offsets"].as_array().unwrap();
            let start = offsets[0].as_u64().unwrap() as usize;
            let end = offsets[1].as_u64().unwrap() as usize;
            self.payload[start..end].fill(0);
        }
        self.write(None);
    }

    pub(crate) fn poison_readout_bias(&mut self) {
        // GroupOutput [group, padded_output] starts with a real output row.
        // Change one affine value without touching any required zero padding.
        let plane = &self.header["language_model.lm_head.biases"];
        assert!(plane["dtype"].as_str() == Some("BF16"), "synthetic readout dtype changed");
        let shape = plane["shape"].as_array().unwrap();
        assert!(shape.len() == 2 && shape[0].as_u64().unwrap() > 0 && shape[1].as_u64().unwrap() > 0);
        let offsets = plane["data_offsets"].as_array().unwrap();
        let start = offsets[0].as_u64().unwrap() as usize;
        self.payload[start..start + 2].copy_from_slice(&0x7fc1_u16.to_le_bytes());
        self.write(None);
    }

    #[cfg(backend = "cpu")]
    fn accepted(&self) -> bool {
        let context = <Cpu as Backend>::Context::new().unwrap();
        let file = File::open(&self.path).unwrap();
        let Ok(loader) = ParameterLoader::<Cpu>::new(&file, &context) else {
            return false;
        };
        Qwen38WeightLoader::<Cpu>::new(context.as_ref(), &self.cfg, &loader).is_ok()
    }
}

fn fill_nonzero(
    plane: &Plane,
    matrices: &BTreeMap<String, Matrix>,
    bytes: &mut [u8],
) {
    let salt = plane.physical.bytes().fold(0_u32, |sum, byte| sum.wrapping_mul(33).wrapping_add(byte as u32));
    if plane.physical_dtype == DataType::U8 {
        let row_bytes = plane.physical_shape[1] as usize;
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte =
                (index as u32).wrapping_mul(37).wrapping_add((index / row_bytes) as u32 * 11).wrapping_add(salt) as u8;
        }
        return;
    }
    let prefix = plane.physical.strip_suffix(".scales").or_else(|| plane.physical.strip_suffix(".biases"));
    if let Some(matrix) = prefix.and_then(|prefix| matrices.values().find(|matrix| matrix.physical == prefix)) {
        let groups = matrix.columns / matrix.group_size;
        let stride = matrix.rows.next_multiple_of(4);
        for row in 0..matrix.rows {
            for group in 0..groups {
                let scale = if matrix.bits == 4 {
                    1.0 / 64.0
                } else {
                    1.0 / 1024.0
                };
                let scale = scale * (1.0 + ((row + group) % 5) as f32 / 16.0);
                let value = if plane.physical.ends_with(".scales") {
                    scale
                } else {
                    -(if matrix.bits == 4 {
                        7.5
                    } else {
                        127.5
                    }) * scale
                };
                let index = if matrix.embedding {
                    row * groups + group
                } else {
                    group * stride + row
                } as usize;
                bytes[index * 2..index * 2 + 2].copy_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
            }
        }
        return;
    }
    for (index, word) in bytes.chunks_exact_mut(plane.physical_dtype.size_in_bytes()).enumerate() {
        let centered = ((index as u32).wrapping_add(salt) % 13) as f32 - 6.0;
        let value = if plane.physical.ends_with(".A_log") {
            -2.0 + centered / 64.0
        } else if plane.physical.ends_with(".dt_bias") {
            0.1 + centered / 256.0
        } else if plane.physical.ends_with(".linear_attn.norm.weight") {
            1.0 + centered / 256.0
        } else if plane.physical.ends_with(".conv1d.weight") {
            centered / 128.0
        } else {
            centered / 256.0
        };
        if plane.physical_dtype == DataType::BF16 {
            word.copy_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
        } else if plane.physical_dtype == DataType::F32 {
            word.copy_from_slice(&value.to_bits().to_le_bytes());
        }
    }
}

#[cfg(backend = "cpu")]
#[stark_test]
fn standard_weight_matrices_preserve_mixed_storage_and_require_full_consumption() {
    let fixture = Fixture::new();
    let context = <Cpu as Backend>::Context::new().unwrap();
    let file = File::open(&fixture.path).unwrap();
    let loader = ParameterLoader::<Cpu>::new(&file, &context).unwrap();
    let weights = Qwen38WeightLoader::<Cpu>::new(context.as_ref(), &fixture.cfg, &loader).unwrap();
    assert!(weights.finish().is_err(), "admission must not mark unread matrix planes consumed");
    let expected = inventory::<Cpu>(&fixture.cfg).unwrap();
    for (name, matrix) in &expected.matrices {
        if matrix.ngram {
            let table = weights.paged_embedding(name, matrix.rows, matrix.columns).unwrap();
            assert!(table.codes.len() > 0 && table.group_size == 32 && table.bits == 4);
        } else if matrix.embedding {
            let _ = weights.embedding(name, matrix.rows, matrix.columns).unwrap();
        } else {
            let binding = weights.binding(&format!("{name}.weight")).unwrap();
            if binding.shape.len() == 3 {
                let bank = weights.bank(name, binding.shape[0], binding.shape[1], matrix.columns).unwrap();
                let quant = bank.matmul_b();
                assert!(quant.group_size() == Some(64) && quant.bits_per_b() == Some(4));
                assert!(bank.values().copyout::<u8>().iter().all(|&byte| byte == 0x12), "expert packed codes changed");
            } else {
                let _ = weights.linear(name, matrix.columns, matrix.rows, DataType::BF16, DataType::BF16).unwrap();
            }
        }
    }
    for (logical, plane) in expected.planes.iter().filter(|(name, plane)| {
        plane.layout == Qwen38TensorLayout::Direct
            && plane.hash.is_none()
            && !name.ends_with(".scales")
            && !name.ends_with(".biases")
    }) {
        if matches!(plane.logical_dtype, Qwen38TensorDType::BF16 | Qwen38TensorDType::F32) {
            let values = weights.control_f32(logical, &plane.logical_shape).unwrap().copyout::<f32>();
            let expected = if plane.logical_dtype == Qwen38TensorDType::F32 {
                0x80000000
            } else {
                0x3f400000
            };
            assert!(
                values.iter().all(|value| value.to_bits() == expected),
                "control promotion changed represented bits"
            );
        }
    }
    weights.finish().unwrap();
}

#[cfg(backend = "cpu")]
#[stark_test]
fn paged_rows_own_the_file_and_reject_reads_outside_the_plane() {
    let context = <Cpu as Backend>::Context::new().unwrap();
    let table = {
        let fixture = Fixture::new();
        let file = File::open(&fixture.path).unwrap();
        let loader = ParameterLoader::<Cpu>::new(&file, &context).unwrap();
        let weights = Qwen38WeightLoader::<Cpu>::new(context.as_ref(), &fixture.cfg, &loader).unwrap();
        weights
            .paged_embedding("model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0", 8, 32)
            .unwrap()
    };
    let mut row = vec![0; table.code_row_bytes];
    table.codes.read_exact_into(0, &mut row).unwrap();
    assert!(row.iter().all(|&byte| byte == 0x12), "paged row changed after loader/file removal");
    assert!(table.codes.read_exact_into(table.codes.len(), &mut [0]).is_err());
    assert!(table.codes.read_exact_into(u64::MAX, &mut [0, 0]).is_err());
    table.codes.read_exact_into(table.codes.len(), &mut []).unwrap();
    assert!(
        table.parameter_row_bytes == 2 && table.scales.shape() == [8, 1] && table.scales.data_type() == DataType::BF16
    );
    for (region, columns, dtype, row_bytes) in [
        (&table.codes, 16, DataType::U8, table.code_row_bytes),
        (&table.scales, 1, DataType::BF16, table.parameter_row_bytes),
        (&table.biases, 1, DataType::BF16, table.parameter_row_bytes),
    ] {
        assert!(
            !region.is_empty()
                && region.shape() == [8, columns]
                && region.data_type() == dtype
                && region.len() == table.rows as u64 * row_bytes as u64,
            "owned paged span no longer matches the consumer row stride"
        );
    }
}

#[cfg(backend = "cpu")]
#[stark_test]
fn strict_admission_rejects_binding_spec_span_padding_and_hash_corruption() {
    let mut fixture = Fixture::new();
    assert!(fixture.accepted());
    let original_header = fixture.header.clone();
    let original_payload = fixture.payload.clone();
    let padded = "language_model.model.layers.0.mlp.shared_expert_gate.biases";
    let offset = fixture.header[padded]["data_offsets"][0].as_u64().unwrap() as usize;
    fixture.payload[offset + 2] = 1;
    fixture.write(None);
    assert!(!fixture.accepted(), "nonzero BF16 affine padding admitted");
    fixture.payload = original_payload.clone();
    fixture.payload.push(0);
    fixture.write(None);
    assert!(!fixture.accepted(), "trailing bytes admitted");
    fixture.payload = original_payload.clone();
    fixture.header[padded]["data_offsets"][0] = serde_json::json!(0);
    fixture.write(None);
    assert!(!fixture.accepted(), "overlapping plane admitted");
    fixture.header = original_header.clone();
    fixture.header["__metadata__"]["unexpected"] = "ignored".into();
    fixture.write(None);
    assert!(!fixture.accepted(), "unbound metadata admitted");
    fixture.header = original_header.clone();
    let key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.spec";
    let mut spec: serde_json::Value =
        serde_json::from_str(fixture.header["__metadata__"][key].as_str().unwrap()).unwrap();
    spec["bits"] = 8.into();
    fixture.header["__metadata__"][key] = spec.to_string().into();
    fixture.write(None);
    assert!(!fixture.accepted(), "contradictory quantization admitted");
    fixture.header = original_header.clone();
    let raw = fixture.header.to_string().replacen("\"dtype\":\"BF16\"", "\"dtype\":\"BF16\",\"dtype\":\"BF16\"", 1);
    fixture.write(Some(raw));
    assert!(!fixture.accepted(), "duplicate nested header key admitted");
    fixture.write(None);
    let hash = "language_model.model.layers.0.ple.ple_embedding.layer_multipliers";
    let offset = fixture.header[hash]["data_offsets"][0].as_u64().unwrap() as usize;
    fixture.payload[offset + 16] ^= 0x80;
    fixture.write(None);
    assert!(!fixture.accepted(), "I64 hash carrier changed unsigned bits");
    fixture.payload = original_payload;
    fixture.write(None);
    let logical = "model.language_model.layers.0.linear_attn.in_proj_a.weight";
    let old = fixture.cfg.tensor_bindings.0[logical].clone();
    fixture.cfg.tensor_bindings.0.get_mut(logical).unwrap().shape[0] += 1;
    assert!(!fixture.accepted(), "logical role shape mismatch admitted");
    fixture.cfg.tensor_bindings.0.insert(logical.into(), old);
    assert!(fixture.accepted(), "restored fixture failed strict admission");
}
