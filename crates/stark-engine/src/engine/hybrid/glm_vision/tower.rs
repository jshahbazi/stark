// SPDX-License-Identifier: Apache-2.0
//! Native GLM5-Next F32 activation vision tower with original BF16/F32 weights.
//!
//! Equations independently implement Transformers GLM5-Next at
//! b6401d6605d16620e0f95b54fb2906f92de7e33f (Hugging Face, Apache-2.0).
//! GGUF names/shapes follow the pinned Unsloth glm5v sidecar. There is no
//! reference Python runtime or whole expanded floating copy of the tower.

use std::{
    fs::OpenOptions,
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::Path,
};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::preprocess::{CHANNEL_PATCH, MAX_PATCHES, PATCH_WIDTH, Patches};
use crate::engine::hybrid::{
    Error, Result,
    math::{rms_norm, silu, softmax},
    weights::Weights,
};

const HIDDEN: usize = 1024;
const HEADS: usize = 16;
const HEAD: usize = HIDDEN / HEADS;
const OUTPUT: usize = 4096;
const EPS: f32 = 1e-5;
const SIDECAR_BYTES: u64 = 1_164_010_176;
const SIDECAR_SHA256: &str = "82b7adb44411dbb298b1c689567804e5ed0790d80f308ed8b5b6b8ac0527d2ac";

pub(in crate::engine::hybrid) struct VisionTower {
    weights: Weights,
}

fn signature(file: &std::fs::File) -> Result<[u64; 7]> {
    let m = file.metadata()?;
    Ok([m.dev(), m.ino(), m.len(), m.mtime() as u64, m.mtime_nsec() as u64, m.ctime() as u64, m.ctime_nsec() as u64])
}

impl VisionTower {
    /// A complete, content-bound sidecar. It is deliberately opened separately
    /// from the decoder split set and keeps its own immutable mapped storage.
    pub fn load(path: &Path) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(path)?;
        let before = signature(&file)?;
        if !file.metadata()?.is_file() || before[2] != SIDECAR_BYTES {
            return Err(Error::Invalid("GLM vision sidecar size/type"));
        }
        let mut digest = Sha256::new();
        let mut buffer = vec![0; 8 * 1024 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        if format!("{:x}", digest.finalize()) != SIDECAR_SHA256 || signature(&file)? != before {
            return Err(Error::Invalid("GLM vision sidecar content changed"));
        }
        let weights = Weights::open(path)?;
        if weights.single_file_signature()? != before {
            return Err(Error::Invalid("GLM vision mapped sidecar identity differs"));
        }
        let current = OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(path)?;
        if signature(&current)? != before {
            return Err(Error::Invalid("GLM vision sidecar changed during loading"));
        }
        validate(&weights)?;
        Ok(Self {
            weights,
        })
    }

    /// Return merged decoder-width rows. Image-owned inputs and projected
    /// features never enter prefix snapshots or a public inspection API.
    pub(super) fn forward(
        &self,
        patches: &Patches,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<f32>> {
        let rows = patches.grid[1] as usize * patches.grid[2] as usize;
        if patches.grid[0] != 1 || rows == 0 || rows > MAX_PATCHES || patches.data.len() != rows * PATCH_WIDTH {
            return Err(Error::Invalid("GLM vision patch geometry"));
        }
        cancelled(should_cancel)?;
        let bias = self.weights.vector("v.patch_embd.bias")?;
        let mut hidden = Vec::with_capacity(rows * HIDDEN);
        for patch in patches.data.chunks_exact(PATCH_WIDTH) {
            let mut temporal = [Vec::with_capacity(3 * CHANNEL_PATCH), Vec::with_capacity(3 * CHANNEL_PATCH)];
            for channel in 0..3 {
                for (t, plane) in temporal.iter_mut().enumerate() {
                    let start = (channel * 2 + t) * CHANNEL_PATCH;
                    plane.extend_from_slice(&patch[start..start + CHANNEL_PATCH]);
                }
            }
            let first = self.weights.floating_patch_matvec("v.patch_embd.weight", &temporal[0])?;
            let second = self.weights.floating_patch_matvec("v.patch_embd.weight.1", &temporal[1])?;
            hidden.extend(first.iter().zip(second).zip(&bias).map(|((&a, b), &bias)| a + b + bias));
            cancelled(should_cancel)?;
        }
        let positions = positions(patches.grid[1] as usize, patches.grid[2] as usize)?;
        for layer in 0..24 {
            cancelled(should_cancel)?;
            let prefix = format!("v.blk.{layer}");
            let norm = normalize_rows(&hidden, HIDDEN, &self.weights.vector(&format!("{prefix}.ln1.weight"))?)?;
            let qkv = linear(
                &self.weights,
                &format!("{prefix}.attn_qkv.weight"),
                Some(&format!("{prefix}.attn_qkv.bias")),
                &norm,
                HIDDEN,
                should_cancel,
            )?;
            let attended = attention(
                &qkv,
                &positions,
                &self.weights.vector(&format!("{prefix}.attn_q_norm.weight"))?,
                &self.weights.vector(&format!("{prefix}.attn_k_norm.weight"))?,
                should_cancel,
            )?;
            let projected = linear(
                &self.weights,
                &format!("{prefix}.attn_out.weight"),
                Some(&format!("{prefix}.attn_out.bias")),
                &attended,
                HIDDEN,
                should_cancel,
            )?;
            add(&mut hidden, &projected)?;
            cancelled(should_cancel)?;
            let norm = normalize_rows(&hidden, HIDDEN, &self.weights.vector(&format!("{prefix}.ln2.weight"))?)?;
            let gate = linear(
                &self.weights,
                &format!("{prefix}.ffn_gate.weight"),
                Some(&format!("{prefix}.ffn_gate.bias")),
                &norm,
                HIDDEN,
                should_cancel,
            )?;
            let up = linear(
                &self.weights,
                &format!("{prefix}.ffn_up.weight"),
                Some(&format!("{prefix}.ffn_up.bias")),
                &norm,
                HIDDEN,
                should_cancel,
            )?;
            let activated = gated(&gate, &up)?;
            let ffn = linear(
                &self.weights,
                &format!("{prefix}.ffn_down.weight"),
                Some(&format!("{prefix}.ffn_down.bias")),
                &activated,
                4096,
                should_cancel,
            )?;
            add(&mut hidden, &ffn)?;
        }
        let normalized = normalize_rows(&hidden, HIDDEN, &self.weights.vector("v.post_ln.weight")?)?;
        let down_bias = self.weights.vector("mm.patch_merger.bias")?;
        let mut downsampled = Vec::with_capacity(rows / 4 * OUTPUT);
        for block in normalized.chunks_exact(4 * HIDDEN) {
            cancelled(should_cancel)?;
            // Conv2d sees [channel, merge_y, merge_x], not four concatenated
            // token rows. Input tokens already use contiguous merge blocks.
            let mut patch = Vec::with_capacity(4 * HIDDEN);
            for channel in 0..HIDDEN {
                for spatial in 0..4 {
                    patch.push(block[spatial * HIDDEN + channel]);
                }
            }
            let mut row = self.weights.floating_patch_matvec("mm.patch_merger.weight", &patch)?;
            add(&mut row, &down_bias)?;
            downsampled.extend(row);
        }
        let projected = linear(&self.weights, "mm.model.fc.weight", None, &downsampled, OUTPUT, should_cancel)?;
        let normalized = layer_norm_rows(
            &projected,
            &self.weights.vector("mm.post_norm.weight")?,
            &self.weights.vector("mm.post_norm.bias")?,
        )?;
        let activated: Vec<f32> = normalized.into_iter().map(gelu).collect();
        let gate = linear(&self.weights, "mm.gate.weight", None, &activated, OUTPUT, should_cancel)?;
        let up = linear(&self.weights, "mm.up.weight", None, &activated, OUTPUT, should_cancel)?;
        cancelled(should_cancel)?;
        let result = linear(&self.weights, "mm.down.weight", None, &gated(&gate, &up)?, 10240, should_cancel)?;
        finite(&result)?;
        Ok(result)
    }
}

fn cancelled(should_cancel: &mut dyn FnMut() -> bool) -> Result<()> {
    if should_cancel() {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

fn finite(values: &[f32]) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        Err(Error::Invalid("GLM vision nonfinite values"))
    } else {
        Ok(())
    }
}

fn add(
    target: &mut [f32],
    value: &[f32],
) -> Result<()> {
    if target.len() != value.len() {
        return Err(Error::Invalid("GLM vision residual shape"));
    }
    for (dst, &src) in target.iter_mut().zip(value) {
        *dst += src;
    }
    finite(target)
}

fn linear(
    weights: &Weights,
    name: &str,
    bias: Option<&str>,
    inputs: &[f32],
    width: usize,
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Vec<f32>> {
    if width == 0 || inputs.len() % width != 0 || inputs.len() / width > MAX_PATCHES {
        return Err(Error::Invalid("GLM vision projection shape"));
    }
    let bias = bias.map(|name| weights.vector(name)).transpose()?;
    let mut result = Vec::new();
    // Eight jobs stay within the existing mapped-matrix weight/workspace caps
    // for the largest projector. Each row retains its native reduction order.
    for chunk in inputs.chunks(width * 8) {
        cancelled(should_cancel)?;
        let jobs: Vec<_> = chunk.chunks_exact(width).map(|row| (name, row)).collect();
        let outputs = weights.matvec_many(&jobs)?;
        cancelled(should_cancel)?;
        for mut row in outputs {
            if let Some(bias) = &bias {
                add(&mut row, bias)?;
            }
            finite(&row)?;
            result.extend(row);
        }
    }
    Ok(result)
}

fn normalize_rows(
    inputs: &[f32],
    width: usize,
    weight: &[f32],
) -> Result<Vec<f32>> {
    if width == 0 || inputs.len() % width != 0 || weight.len() != width {
        return Err(Error::Invalid("GLM vision normalization shape"));
    }
    let mut result = Vec::with_capacity(inputs.len());
    for row in inputs.chunks_exact(width) {
        result.extend(rms_norm(row, weight, EPS)?);
    }
    Ok(result)
}

fn positions(
    height: usize,
    width: usize,
) -> Result<Vec<[usize; 2]>> {
    if height == 0
        || width == 0
        || height % 2 != 0
        || width % 2 != 0
        || height.checked_mul(width).is_none_or(|n| n > MAX_PATCHES)
    {
        return Err(Error::Invalid("GLM vision rotary grid"));
    }
    let mut result = Vec::with_capacity(height * width);
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    result.push([y + dy, x + dx]);
                }
            }
        }
    }
    Ok(result)
}

fn rotary(
    values: &mut [f32],
    position: [usize; 2],
) -> Result<()> {
    if values.len() != HEAD {
        return Err(Error::Invalid("GLM vision rotary head"));
    }
    let old = values.to_vec();
    for axis in 0..2 {
        for index in 0..HEAD / 4 {
            let angle = position[axis] as f32 / 10000.0f32.powf(index as f32 / (HEAD / 4) as f32);
            let (sin, cos) = angle.sin_cos();
            let i = axis * HEAD / 4 + index;
            values[i] = old[i] * cos - old[i + HEAD / 2] * sin;
            values[i + HEAD / 2] = old[i + HEAD / 2] * cos + old[i] * sin;
        }
    }
    finite(values)
}

fn attention(
    qkv: &[f32],
    positions: &[[usize; 2]],
    query_norm: &[f32],
    key_norm: &[f32],
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Vec<f32>> {
    let rows = positions.len();
    if rows == 0
        || rows > MAX_PATCHES
        || qkv.len() != rows * 3 * HIDDEN
        || query_norm.len() != HEAD
        || key_norm.len() != HEAD
    {
        return Err(Error::Invalid("GLM vision attention shape"));
    }
    let (mut queries, mut keys, mut values) =
        (Vec::with_capacity(rows * HIDDEN), Vec::with_capacity(rows * HIDDEN), Vec::with_capacity(rows * HIDDEN));
    for (row, &position) in qkv.chunks_exact(3 * HIDDEN).zip(positions) {
        for head in 0..HEADS {
            let mut query = rms_norm(&row[head * HEAD..(head + 1) * HEAD], query_norm, EPS)?;
            let mut key = rms_norm(&row[HIDDEN + head * HEAD..HIDDEN + (head + 1) * HEAD], key_norm, EPS)?;
            rotary(&mut query, position)?;
            rotary(&mut key, position)?;
            queries.extend(query);
            keys.extend(key);
        }
        values.extend_from_slice(&row[2 * HIDDEN..]);
    }
    let mut output = vec![0.0; rows * HIDDEN];
    for row in 0..rows {
        cancelled(should_cancel)?;
        for head in 0..HEADS {
            let query = &queries[row * HIDDEN + head * HEAD..row * HIDDEN + (head + 1) * HEAD];
            let scores: Vec<f32> = (0..rows)
                .map(|source| {
                    query
                        .iter()
                        .zip(&keys[source * HIDDEN + head * HEAD..source * HIDDEN + (head + 1) * HEAD])
                        .map(|(&q, &k)| q * k)
                        .sum::<f32>()
                        * 0.125
                })
                .collect();
            let probabilities = softmax(&scores)?;
            for (source, &probability) in probabilities.iter().enumerate() {
                for lane in 0..HEAD {
                    output[row * HIDDEN + head * HEAD + lane] +=
                        probability * values[source * HIDDEN + head * HEAD + lane];
                }
            }
        }
    }
    finite(&output)?;
    Ok(output)
}

fn gated(
    gate: &[f32],
    up: &[f32],
) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(Error::Invalid("GLM vision gated shape"));
    }
    finite(gate)?;
    finite(up)?;
    Ok(gate.iter().zip(up).map(|(&gate, &up)| silu(gate.min(10.0)) * up.clamp(-10.0, 10.0)).collect())
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn layer_norm_rows(
    inputs: &[f32],
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>> {
    if weight.len() != OUTPUT || bias.len() != OUTPUT || inputs.len() % OUTPUT != 0 {
        return Err(Error::Invalid("GLM vision layer norm shape"));
    }
    let mut output = Vec::with_capacity(inputs.len());
    for row in inputs.chunks_exact(OUTPUT) {
        let mean = row.iter().sum::<f32>() / OUTPUT as f32;
        let variance = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / OUTPUT as f32;
        let scale = (variance + EPS).sqrt().recip();
        output.extend(row.iter().zip(weight).zip(bias).map(|((&v, &w), &b)| (v - mean) * scale * w + b));
    }
    finite(&output)?;
    Ok(output)
}

fn validate(weights: &Weights) -> Result<()> {
    let metadata = weights.metadata();
    if metadata.get("general.architecture").and_then(Value::as_str) != Some("clip")
        || metadata.get("clip.projector_type").and_then(Value::as_str) != Some("glm5v")
        || weights.names().len() != 348
    {
        return Err(Error::Invalid("GLM vision sidecar architecture/inventory"));
    }
    for (name, expected) in [
        ("clip.vision.block_count", 24),
        ("clip.vision.embedding_length", 1024),
        ("clip.vision.attention.head_count", 16),
        ("clip.vision.patch_size", 14),
        ("clip.vision.spatial_merge_size", 2),
        ("clip.vision.projection_dim", 4096),
        ("clip.vision.image_min_pixels", 12544),
        ("clip.vision.image_max_pixels", 6272000),
    ] {
        if metadata.get(name).and_then(Value::as_u64) != Some(expected) {
            return Err(Error::Invalid("GLM vision sidecar metadata"));
        }
    }
    if metadata.get("clip.vision.swiglu_clamp").and_then(Value::as_f64) != Some(10.0) {
        return Err(Error::Invalid("GLM vision sidecar clamp"));
    }
    let mut expected_names = std::collections::BTreeSet::new();
    let mut check = |name: String, shape: &[usize], kind: u32| -> Result<()> {
        if weights.shape(&name)? != shape || weights.floating_tensor_kind(&name)? != kind {
            return Err(Error::Invalid("GLM vision tensor shape/precision"));
        }
        expected_names.insert(name);
        Ok(())
    };
    for layer in 0..24 {
        let p = format!("v.blk.{layer}");
        for name in ["attn_k_norm", "attn_q_norm"] {
            check(format!("{p}.{name}.weight"), &[64], 0)?;
        }
        for name in ["ln1", "ln2"] {
            check(format!("{p}.{name}.weight"), &[1024], 0)?;
        }
        for (name, rows, cols) in [
            ("attn_qkv", 3072, 1024),
            ("attn_out", 1024, 1024),
            ("ffn_gate", 4096, 1024),
            ("ffn_up", 4096, 1024),
            ("ffn_down", 1024, 4096),
        ] {
            check(format!("{p}.{name}.weight"), &[rows, cols], 30)?;
            check(format!("{p}.{name}.bias"), &[rows], 0)?;
        }
    }
    for name in ["v.patch_embd.weight", "v.patch_embd.weight.1"] {
        check(name.into(), &[1024, 3, 14, 14], 0)?;
    }
    check("v.patch_embd.bias".into(), &[1024], 0)?;
    check("v.post_ln.weight".into(), &[1024], 0)?;
    check("mm.patch_merger.weight".into(), &[4096, 1024, 2, 2], 0)?;
    check("mm.patch_merger.bias".into(), &[4096], 0)?;
    for name in ["mm.post_norm.weight", "mm.post_norm.bias"] {
        check(name.into(), &[4096], 0)?;
    }
    for (name, rows, cols) in [
        ("mm.model.fc.weight", 4096, 4096),
        ("mm.gate.weight", 10240, 4096),
        ("mm.up.weight", 10240, 4096),
        ("mm.down.weight", 4096, 10240),
    ] {
        check(name.into(), &[rows, cols], 30)?;
    }
    if expected_names != weights.names().into_iter().collect() {
        return Err(Error::Invalid("GLM vision tensor inventory"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn glm_rotary_merge_coordinates_and_norm_preservation() {
        let coords = positions(4, 4).unwrap();
        assert_eq!(&coords[..8], &[[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]]);
        assert!(positions(3, 4).is_err());
        let mut values: Vec<f32> = (0..HEAD).map(|index| index as f32 / 32.0 - 1.0).collect();
        let original = values.clone();
        rotary(&mut values, [0, 0]).unwrap();
        assert_eq!(values, original);
        let original_norm = values.iter().map(|v| v * v).sum::<f32>();
        rotary(&mut values, [17, 31]).unwrap();
        let rotated_norm = values.iter().map(|v| v * v).sum::<f32>();
        assert!((rotated_norm - original_norm).abs() < 1e-4);
    }

    #[stark_test]
    fn glm_attention_preserves_constant_values_and_honors_cancellation() {
        let mut qkv = vec![0.0; 4 * 3 * HIDDEN];
        for (index, row) in qkv.chunks_exact_mut(3 * HIDDEN).enumerate() {
            for lane in 0..2 * HIDDEN {
                row[lane] = (lane % 17) as f32 * 0.01 + index as f32;
            }
            for lane in 0..HIDDEN {
                row[2 * HIDDEN + lane] = (lane % 13) as f32 - 3.0;
            }
        }
        let output = attention(&qkv, &positions(2, 2).unwrap(), &[1.0; HEAD], &[1.0; HEAD], &mut || false).unwrap();
        for row in output.chunks_exact(HIDDEN) {
            assert!(row.iter().enumerate().all(|(lane, &value)| (value - ((lane % 13) as f32 - 3.0)).abs() < 1e-5));
        }
        assert!(matches!(
            attention(&qkv, &positions(2, 2).unwrap(), &[1.0; HEAD], &[1.0; HEAD], &mut || true),
            Err(Error::Cancelled)
        ));
    }

    #[stark_test]
    fn glm_projector_norm_gelu_and_clamp_controls() {
        let normalized = layer_norm_rows(&vec![7.0; OUTPUT], &vec![2.0; OUTPUT], &vec![3.0; OUTPUT]).unwrap();
        assert!(normalized.iter().all(|&value| value == 3.0));
        let activated = gated(&[20.0, 10.0, -20.0], &[20.0, 10.0, -20.0]).unwrap();
        assert_eq!(activated[0], activated[1]);
        assert!(activated[2] > 0.0 && activated[2] < 1e-5);
        assert!(gated(&[f32::NAN], &[1.0]).is_err());
        assert!(gelu(0.0) == 0.0);
        assert!((gelu(1.0) - 0.8413447).abs() < 1e-6);
    }

    #[stark_test]
    #[ignore]
    fn glm_vision_attention_matches_private_torch_f32_component_reference() {
        let path = std::env::var_os("STARK_GLM_VISION_ATTENTION_REFERENCE").expect("GLM attention reference required");
        let bytes = std::fs::read(Path::new(&path)).unwrap();
        assert!(bytes.len() <= 2 * 1024 * 1024);
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["transformers_revision"].as_str(), Some("b6401d6605d16620e0f95b54fb2906f92de7e33f"));
        let values = |name: &str| -> Vec<f32> {
            value[name].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()
        };
        let actual = attention(
            &values("qkv"),
            &positions(2, 2).unwrap(),
            &values("query_weight"),
            &values("key_weight"),
            &mut || false,
        )
        .unwrap();
        let expected = values("attention_output");
        assert_eq!(actual.len(), expected.len());
        let maximum = actual.iter().zip(&expected).map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);
        let relative = (actual.iter().zip(&expected).map(|(&a, &b)| (a - b) * (a - b)).sum::<f32>()
            / expected.iter().map(|v| v * v).sum::<f32>())
        .sqrt();
        assert!(maximum <= 1e-3 && relative <= 1e-3, "GLM F32 attention component reference gate failed");
    }

    #[stark_test]
    #[ignore]
    fn glm_actual_tower_matches_private_torch_f32_reference() {
        let reference = std::env::var_os("STARK_GLM_VISION_TOWER_REFERENCE").expect("GLM tower reference required");
        let reference = Path::new(&reference);
        assert!(std::fs::metadata(reference).unwrap().len() <= 8 * 1024 * 1024);
        let value: Value = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
        assert_eq!(value["source_revision"].as_str(), Some("b6401d6605d16620e0f95b54fb2906f92de7e33f"));
        assert_eq!(value["sidecar_sha256"].as_str(), Some(SIDECAR_SHA256));
        let values = |name: &str| -> Vec<f32> {
            value[name].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()
        };
        let grid: Vec<u32> = value["grid"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(grid, [1, 4, 4]);
        let patches = Patches {
            data: values("patches").into_boxed_slice(),
            grid: [1, 4, 4],
            dimensions: [56, 56],
        };
        let path = std::env::var_os("STARK_GLM_VISION_SIDECAR").expect("pinned GLM sidecar path required");
        let tower = VisionTower::load(Path::new(&path)).unwrap();
        let output = tower.forward(&patches, &mut || false).unwrap();
        let expected = values("output");
        assert_eq!(output.len(), expected.len());
        let maximum = output.iter().zip(&expected).map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);
        let relative = (output.iter().zip(&expected).map(|(&a, &b)| (a - b) * (a - b)).sum::<f32>()
            / expected.iter().map(|v| v * v).sum::<f32>())
        .sqrt();
        println!(
            "{}",
            serde_json::json!({"glm_actual_tower_f32_reference":true,"input_rows":16,"output_values":output.len(),"maximum_absolute_error":maximum,"relative_rms_error":relative})
        );
        assert!(maximum <= 1e-3 && relative <= 1e-3, "GLM actual F32 tower reference gate failed");
    }

    #[stark_test]
    #[ignore]
    fn glm_pinned_actual_tower_has_finite_decoder_width_features() {
        let path = std::env::var_os("STARK_GLM_VISION_SIDECAR").expect("pinned GLM sidecar path required");
        let tower = VisionTower::load(Path::new(&path)).unwrap();
        let mut encoded = Vec::new();
        use image::ImageEncoder;
        image::codecs::png::PngEncoder::new(&mut encoded)
            .write_image(&[37, 109, 211].repeat(32 * 32), 32, 32, image::ExtendedColorType::Rgb8)
            .unwrap();
        let patches = super::super::preprocess::prepare(&encoded).unwrap();
        let output = tower.forward(&patches, &mut || false).unwrap();
        assert_eq!(output.len(), 16 * OUTPUT);
        assert!(output.iter().all(|v| v.is_finite()));
        assert!(matches!(tower.forward(&patches, &mut || true), Err(Error::Cancelled)));
    }
}
