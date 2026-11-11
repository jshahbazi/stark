//! Stark's ignored source-bound tower replay; no production API is added.

use std::{fs, path::Path};

use half::bf16;
use serde_json::{Value, json};

use super::*;
use crate::{
    qwen4exp::config::Qwen4ExpConfig,
    safetensors::Checkpoint,
    sha256::{hex, sha256},
    weights::Loader,
};

fn checked_file(root: &Path, record: &Value, cap: usize) -> Vec<u8> {
    let filename =
        record["file"].as_str().expect("private reference filename unavailable");
    assert!(!filename.contains(['/', '\\']) && filename != "." && filename != "..");
    let path = root.join(filename);
    assert!(
        fs::metadata(&path).expect("private reference metadata unavailable").len()
            <= cap as u64
    );
    let data = fs::read(path).expect("private reference unavailable");
    assert_eq!(
        hex(&sha256(&data)),
        record["sha256"].as_str().expect("private reference digest unavailable")
    );
    data
}

fn checked_tensor(root: &Path, record: &Value) -> Vec<bf16> {
    let bytes = checked_file(root, record, 2 * 256 * 4608);
    let shape = record["shape"].as_array().expect("private tensor shape unavailable");
    let elements = shape
        .iter()
        .try_fold(1_usize, |count, value| {
            count.checked_mul(usize::try_from(value.as_u64()?).ok()?)
        })
        .expect("private tensor shape invalid");
    assert_eq!(elements.checked_mul(2), Some(bytes.len()));
    bytes
        .chunks_exact(2)
        .map(|value| bf16::from_bits(u16::from_le_bytes([value[0], value[1]])))
        .collect()
}

fn check_weight(record: &Value, name: &str, tensor: &Tensor) {
    assert_eq!(tensor.dtype(), DType::BF16);
    assert_eq!(
        hex(&sha256(tensor.raw_bytes())),
        record[name].as_str().expect("private tensor identity missing")
    );
}

fn numerical_metric(actual: &[f32], expected: &[bf16]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let (mut error, mut reference_norm, mut actual_norm, mut dot) =
        (0_f64, 0_f64, 0_f64, 0_f64);
    for (&a, &b) in actual.iter().zip(expected) {
        let (a, b) = (f64::from(a), f64::from(f32::from(b)));
        assert!(a.is_finite() && b.is_finite());
        error += (a - b).powi(2);
        reference_norm += b * b;
        actual_norm += a * a;
        dot += a * b;
    }
    ((error / reference_norm).sqrt(), dot / (actual_norm * reference_norm).sqrt())
}

fn checked_f32(root: &Path, record: &Value, shape: &[usize]) -> Vec<f32> {
    assert_eq!(record["shape"], json!(shape));
    let bytes = checked_file(root, record, 256 * 72 * 4);
    assert_eq!(bytes.len(), shape.iter().product::<usize>() * 4);
    let values = bytes
        .chunks_exact(4)
        .map(|part| f32::from_le_bytes(part.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(values.iter().all(|value| value.is_finite()));
    values
}

fn attention_order_metrics(
    context: &MetalContext,
    input: &Tensor,
    folder: &Path,
    record: &Value,
    expected: &[bf16],
    ordinary: &[f32],
    n: usize,
) {
    use crate::kernels::vision::{
        attention_order_pv, attention_order_qk, attention_order_scale,
        attention_order_softmax,
    };
    let qk_expected = checked_tensor(folder, &record["stages"]["qk"]);
    let scaled_expected = checked_tensor(folder, &record["stages"]["scaled"]);
    let probabilities_expected =
        checked_tensor(folder, &record["stages"]["probabilities"]);
    let shape = [16, n, n];
    let z =
        |dimensions: &[usize]| Tensor::zeros(context, dimensions, DType::BF16).unwrap();
    let source = |values: &[bf16]| {
        Tensor::from_bytes(context, bytemuck::cast_slice(values), &shape, DType::BF16)
            .unwrap()
    };
    let qk = z(&shape);
    let scaled = z(&shape);
    let probabilities = z(&shape);
    let output = z(&[n, 1152]);
    let scale = (72_f32).powf(-0.5);
    let pass = context.begin().unwrap();
    attention_order_qk(context, &pass, input, &qk, &scaled, n, scale).unwrap();
    attention_order_softmax(context, &pass, &scaled, &probabilities, n).unwrap();
    attention_order_pv(context, &pass, input, &probabilities, &output, n).unwrap();
    pass.commit_wait().unwrap();
    let qk_values = qk.to_f32().unwrap();
    let scaled_values = scaled.to_f32().unwrap();
    let probabilities_values = probabilities.to_f32().unwrap();
    let output_values = output.to_f32().unwrap();
    let source_qk = source(&qk_expected);
    let source_scaled = source(&scaled_expected);
    let source_probabilities = source(&probabilities_expected);
    let scaled_only = z(&shape);
    let probabilities_only = z(&shape);
    let pv_only = z(&[n, 1152]);
    let pass = context.begin().unwrap();
    attention_order_scale(context, &pass, &source_qk, &scaled_only, n, scale).unwrap();
    attention_order_softmax(context, &pass, &source_scaled, &probabilities_only, n)
        .unwrap();
    attention_order_pv(context, &pass, input, &source_probabilities, &pv_only, n)
        .unwrap();
    pass.commit_wait().unwrap();
    let metric = |index: usize, actual: &[f32], reference: &[bf16]| {
        let (rms, cosine) = numerical_metric(actual, reference);
        let mismatches = actual
            .iter()
            .zip(reference)
            .filter(|(a, b)| bf16::from_f32(**a).to_bits() != b.to_bits())
            .count();
        json!({"stage_index":index,"elements":actual.len(),"mismatch_count":mismatches,"relative_rms":rms,"cosine":cosine})
    };
    let metrics = vec![
        metric(0, &qk_values, &qk_expected),
        metric(1, &scaled_values, &scaled_expected),
        metric(2, &scaled_only.to_f32().unwrap(), &scaled_expected),
        metric(3, &probabilities_only.to_f32().unwrap(), &probabilities_expected),
        metric(4, &pv_only.to_f32().unwrap(), expected),
        metric(5, &probabilities_values, &probabilities_expected),
        metric(6, &output_values, expected),
    ];
    // Same reference graph with per-stage commits/readbacks must preserve all
    // intermediate and final bits; ordinary production's observer stays intact.
    let pass = context.begin().unwrap();
    attention_order_qk(context, &pass, input, &qk, &scaled, n, scale).unwrap();
    pass.commit_wait().unwrap();
    let mismatch = |a: &[f32], b: &[f32]| {
        a.iter().zip(b).filter(|(a, b)| a.to_bits() != b.to_bits()).count()
    };
    let mut observer = mismatch(&qk.to_f32().unwrap(), &qk_values)
        + mismatch(&scaled.to_f32().unwrap(), &scaled_values);
    let pass = context.begin().unwrap();
    attention_order_softmax(context, &pass, &scaled, &probabilities, n).unwrap();
    pass.commit_wait().unwrap();
    observer += mismatch(&probabilities.to_f32().unwrap(), &probabilities_values);
    let pass = context.begin().unwrap();
    attention_order_pv(context, &pass, input, &probabilities, &output, n).unwrap();
    pass.commit_wait().unwrap();
    observer += mismatch(&output.to_f32().unwrap(), &output_values);
    let ordinary_mismatches = ordinary
        .iter()
        .zip(expected)
        .filter(|(a, b)| bf16::from_f32(**a).to_bits() != b.to_bits())
        .count();
    println!(
        "{}",
        json!({"private_retained_attention_order_metrics":{"patches":n,"stages":metrics,"observer_mismatch_count":observer,"ordinary_mismatch_count":ordinary_mismatches}})
    );
    assert_eq!(
        observer, 0,
        "private attention reference stage readbacks perturb ordinary execution"
    );
    assert!(
        metrics.iter().all(|metric| metric["mismatch_count"] == 0),
        "private attention order witness exceeds exact source gate"
    );
}

fn rotary_source_metrics(
    context: &MetalContext,
    input: &[bf16],
    expected: &[bf16],
    safe: &[f32],
    case: &Value,
    n: usize,
    gw: usize,
) {
    let Some(root) = std::env::var_os("STARK_LILY_ROTARY_REFERENCE_PATH") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    assert!(fs::metadata(root.join("manifest.json")).unwrap().len() <= 65536);
    let manifest: Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["kind"], "RetainedLilyRotarySourceReference");
    assert_eq!(manifest["torch_git"], "5c4886908584029761b579af026dcfb627c84070");
    assert_eq!(
        manifest["source_sha256"],
        "e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df"
    );
    let record = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|record| record["name"] == case["name"])
        .unwrap();
    assert_eq!(record["grid"], case["grid"]);
    let folder = root.join(case["name"].as_str().unwrap());
    let inv_expected = checked_f32(&root, &manifest["inv_freq"], &[18]);
    let angles = checked_f32(&folder, &record["angles"], &[n, 36]);
    let compact = |values: Vec<f32>| {
        values
            .chunks_exact(72)
            .flat_map(|row| row[..36].iter().copied())
            .collect::<Vec<_>>()
    };
    let cos_expected = compact(checked_f32(&folder, &record["cos"], &[n, 72]));
    let sin_expected = compact(checked_f32(&folder, &record["sin"], &[n, 72]));
    let tensor = |values: &[f32], shape: &[usize]| {
        Tensor::from_bytes(context, bytemuck::cast_slice(values), shape, DType::F32)
            .unwrap()
    };
    let source_angles = tensor(&angles, &[n, 36]);
    let source_cos = tensor(&cos_expected, &[n, 36]);
    let source_sin = tensor(&sin_expected, &[n, 36]);
    let stages = Tensor::zeros(context, &[5, n, 36], DType::F32).unwrap();
    let inv = Tensor::zeros(context, &[18], DType::F32).unwrap();
    let pass = context.begin().unwrap();
    crate::kernels::vision::rotary_stages_reference(
        context,
        &pass,
        &source_angles,
        &stages,
        &inv,
        n,
        gw,
    )
    .unwrap();
    pass.commit_wait().unwrap();
    let values = stages.to_f32().unwrap();
    let inverse = inv.to_f32().unwrap();
    let rust_inverse = (0..18)
        .map(|index| 1.0_f32 / 10000_f32.powf((2 * index) as f32 / 36.0))
        .collect::<Vec<_>>();
    let mut host_angles = Vec::with_capacity(n * 36);
    for row in 0..n {
        let ih = (row / 2) % 2;
        let iw = row % 2;
        let bh = row / (4 * (gw / 2));
        let bw = (row / 4) % (gw / 2);
        for index in 0..36 {
            let position = if index < 18 { 2 * bh + ih } else { 2 * bw + iw };
            host_angles.push(position as f32 * inv_expected[index % 18]);
        }
    }
    let (sleef_cos, sleef_sin) = super::source_math::vision_sin_cos(&host_angles);
    let (production_cos, production_sin) =
        super::rotary_coefficients(n / gw, gw).unwrap();
    let plane = n * 36;
    let pairs = [
        (&inverse[..], &inv_expected[..]),
        (&values[..plane], &angles[..]),
        (&values[plane..2 * plane], &cos_expected[..]),
        (&values[2 * plane..3 * plane], &sin_expected[..]),
        (&values[3 * plane..4 * plane], &cos_expected[..]),
        (&values[4 * plane..5 * plane], &sin_expected[..]),
        (&rust_inverse[..], &inv_expected[..]),
        (&host_angles[..], &angles[..]),
        (&sleef_cos[..], &cos_expected[..]),
        (&sleef_sin[..], &sin_expected[..]),
        (&production_cos[..], &cos_expected[..]),
        (&production_sin[..], &sin_expected[..]),
    ];
    let stage_metrics = pairs.iter().enumerate().map(|(index,(actual,reference))| {
        let mut error=0_f64; let mut norm=0_f64; let mut mismatches=0; let mut ulps=0;
        for (a,b) in actual.iter().zip(reference.iter()) {
            assert!(a.is_finite() && b.is_finite());
            error += (f64::from(*a)-f64::from(*b)).powi(2); norm += f64::from(*b).powi(2);
            mismatches += usize::from(a.to_bits()!=b.to_bits()); ulps=ulps.max(a.to_bits().abs_diff(b.to_bits()));
        }
        json!({"stage_index":index,"elements":actual.len(),"mismatch_count":mismatches,"max_ulp":ulps,"relative_rms":(error/norm.max(f64::MIN_POSITIVE)).sqrt()})
    }).collect::<Vec<_>>();
    let source_applied = Tensor::from_bytes(
        context,
        bytemuck::cast_slice(input),
        &[n, 3456],
        DType::BF16,
    )
    .unwrap();
    let native_applied = Tensor::from_bytes(
        context,
        bytemuck::cast_slice(input),
        &[n, 3456],
        DType::BF16,
    )
    .unwrap();
    let exact_angle_applied = Tensor::from_bytes(
        context,
        bytemuck::cast_slice(input),
        &[n, 3456],
        DType::BF16,
    )
    .unwrap();
    let sleef_applied = Tensor::from_bytes(
        context,
        bytemuck::cast_slice(input),
        &[n, 3456],
        DType::BF16,
    )
    .unwrap();
    let sleef_cos_tensor = tensor(&sleef_cos, &[n, 36]);
    let sleef_sin_tensor = tensor(&sleef_sin, &[n, 36]);
    let production_applied = Tensor::from_bytes(
        context,
        bytemuck::cast_slice(input),
        &[n, 3456],
        DType::BF16,
    )
    .unwrap();
    let production_cos_tensor = tensor(&production_cos, &[n, 36]);
    let production_sin_tensor = tensor(&production_sin, &[n, 36]);
    let pass = context.begin().unwrap();
    crate::kernels::vision::rotary_apply_reference(
        context,
        &pass,
        &source_applied,
        &source_cos,
        &source_sin,
        n,
    )
    .unwrap();
    crate::kernels::vision::rotary_apply_reference(
        context,
        &pass,
        &native_applied,
        &stages.view(plane, &[n, 36]).unwrap(),
        &stages.view(2 * plane, &[n, 36]).unwrap(),
        n,
    )
    .unwrap();
    crate::kernels::vision::rotary_apply_reference(
        context,
        &pass,
        &exact_angle_applied,
        &stages.view(3 * plane, &[n, 36]).unwrap(),
        &stages.view(4 * plane, &[n, 36]).unwrap(),
        n,
    )
    .unwrap();
    crate::kernels::vision::rotary_apply_reference(
        context,
        &pass,
        &sleef_applied,
        &sleef_cos_tensor,
        &sleef_sin_tensor,
        n,
    )
    .unwrap();
    crate::kernels::vision::qkv_rope_coefficients_bf16(
        context,
        &pass,
        &production_applied,
        &production_cos_tensor,
        &production_sin_tensor,
        n / gw,
        gw,
    )
    .unwrap();
    pass.commit_wait().unwrap();
    let production_applied = production_applied.to_f32().unwrap();
    let sleef_applied = sleef_applied.to_f32().unwrap();
    let exact_angle_applied = exact_angle_applied.to_f32().unwrap();
    let source_applied = source_applied.to_f32().unwrap();
    let native_applied = native_applied.to_f32().unwrap();
    let source_mismatches = source_applied
        .iter()
        .zip(expected)
        .filter(|(a, b)| bf16::from_f32(**a).to_bits() != b.to_bits())
        .count();
    let observer_mismatches = native_applied
        .iter()
        .zip(safe)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let exact_angle_mismatches = exact_angle_applied
        .iter()
        .zip(expected)
        .filter(|(a, b)| bf16::from_f32(**a).to_bits() != b.to_bits())
        .count();
    let sleef_mismatches = sleef_applied
        .iter()
        .zip(expected)
        .filter(|(a, b)| bf16::from_f32(**a).to_bits() != b.to_bits())
        .count();
    let production_mismatches = production_applied
        .iter()
        .zip(expected)
        .filter(|(a, b)| bf16::from_f32(**a).to_bits() != b.to_bits())
        .count();
    println!(
        "{}",
        json!({"private_retained_rotary_stage_metrics":{"patches":n,"stages":stage_metrics,"source_application_mismatch_count":source_mismatches,"native_application_observer_mismatch_count":observer_mismatches,"source_angle_native_trig_mismatch_count":exact_angle_mismatches,"sleef_application_mismatch_count":sleef_mismatches,"production_mismatch_count":production_mismatches}})
    );
    assert_eq!(
        source_mismatches, 0,
        "private rotary source-angle application exceeds exact gate"
    );
    assert_eq!(
        observer_mismatches, 0,
        "private rotary stages perturb the safe witness"
    );
    assert_eq!(
        production_mismatches, 0,
        "private vision rotary exceeds exact source gate"
    );
    assert!(
        stage_metrics[10..12].iter().all(|metric| metric["mismatch_count"] == 0),
        "private vision rotary coefficients exceed exact source gate"
    );
}

fn gelu_domain_metrics(context: &MetalContext) {
    let Some(root) = std::env::var_os("STARK_LILY_GELU_REFERENCE_PATH") else { return };
    let root = std::path::PathBuf::from(root);
    assert!(fs::metadata(root.join("manifest.json")).unwrap().len() <= 65536);
    let manifest: Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["kind"], "RetainedLilyGeluDomainReference");
    assert_eq!(manifest["torch_git"], "5c4886908584029761b579af026dcfb627c84070");
    assert_eq!(manifest["finite_inputs"], 65280);
    assert_eq!(manifest["input"]["shape"], json!([255, 256]));
    let input = checked_tensor(&root, &manifest["input"]);
    let expected_bits = (0..=u16::MAX).filter(|value| value & 0x7f80 != 0x7f80);
    assert!(
        input.iter().map(|value| value.to_bits()).eq(expected_bits),
        "private GELU domain input identity"
    );
    let mut metrics = Vec::new();
    for (index, (name, kind)) in
        [("erf", Gelu::Erf), ("tanh", Gelu::Tanh)].into_iter().enumerate()
    {
        assert_eq!(manifest[name]["shape"], json!([255, 256]));
        let expected = checked_tensor(&root, &manifest[name]);
        let values = Tensor::from_bytes(
            context,
            bytemuck::cast_slice(&input),
            &[255, 256],
            DType::BF16,
        )
        .unwrap();
        let pass = context.begin().unwrap();
        gelu_vision_bf16(context, &pass, &values, kind)
            .unwrap_or_else(|_| panic!("private GELU domain execution failed"));
        pass.commit_wait().unwrap();
        let actual = values.to_f32().unwrap();
        let mismatches = actual
            .iter()
            .zip(expected)
            .filter(|(a, b)| bf16::from_f32(**a).to_bits() != b.to_bits())
            .count();
        let nonfinite = actual.iter().filter(|value| !value.is_finite()).count();
        metrics.push(json!({"kind_index":index,"elements":actual.len(),"mismatch_count":mismatches,"nonfinite_count":nonfinite}));
    }
    println!("{}", json!({"private_retained_gelu_domain_metrics":metrics}));
    assert!(
        metrics
            .iter()
            .all(|metric| metric["mismatch_count"] == 0
                && metric["nonfinite_count"] == 0),
        "private GELU domain exceeds exact source gate"
    );
}

/// Bounded first-block sequence, using the same scratch layout and production
/// kernels. Readbacks are aggregate comparisons only. The caller separately
/// requires the final merged bits to equal the ordinary one-pass graph.
fn first_block_metrics(
    context: &MetalContext,
    vision: &super::super::config::VisionConfig,
    weights: &super::super::vision_weights::VisionWeights,
    scratch: &mut Option<VisionScratch>,
    folder: &Path,
    case: &Value,
    pixels: &[f32],
    gh: usize,
    gw: usize,
) -> Option<Vec<f32>> {
    let records = case["operators"].as_array()?;
    let expected = |operator: &str, field: &str| {
        let record = records
            .iter()
            .find(|record| record["operator"] == operator)
            .expect("private sequence reference missing");
        checked_tensor(folder, &record[field])
    };
    let (n, hid, merge_dim) = (gh * gw, vision.hidden_size, vision.merge_dim());
    VisionTower::ensure_scratch(context, scratch, vision, n).unwrap();
    let s = scratch.as_ref().unwrap();
    let px = s.pixels.view(0, &[n, vision.patch_dim()]).unwrap();
    let x = s.x.view(0, &[n, hid]).unwrap();
    let h = s.h.view(0, &[n, hid]).unwrap();
    let qkv = s.qkv.view(0, &[n, 3 * hid]).unwrap();
    let attn = s.attn.view(0, &[n, hid]).unwrap();
    let tmp = s.tmp.view(0, &[n, hid]).unwrap();
    let mlp = s.mlp.view(0, &[n, vision.intermediate_size]).unwrap();
    let h4 = s.h.view(0, &[n / 4, merge_dim]).unwrap();
    let mlp4 = s.mlp.view(0, &[n / 4, merge_dim]).unwrap();
    let merged = s.merged.view(0, &[n / 4, vision.out_hidden_size]).unwrap();
    let cast = pixels.iter().map(|value| bf16::from_f32(*value)).collect::<Vec<_>>();
    px.write_bytes(bytemuck::cast_slice(&cast)).unwrap();
    let rotary = prepare_rotary(s, gh, gw).unwrap();
    let block = &weights.blocks[0];
    let mut metrics = Vec::new();
    macro_rules! stage {
        ($tensor:expr, $operator:expr, $field:expr, $dispatch:expr) => {{
            let pass = context.begin().unwrap();
            ($dispatch)(&pass).unwrap_or_else(|_: anyhow::Error| panic!("private sequence execution failed"));
            pass.commit_wait().unwrap_or_else(|_| panic!("private sequence synchronization failed"));
            let actual = $tensor.to_f32().unwrap();
            let (rms, cosine) = numerical_metric(&actual, &expected($operator, $field));
            metrics.push(json!({"stage_index":metrics.len(),"elements":actual.len(),"relative_rms":rms,"cosine":cosine}));
        }};
    }
    stage!(x, "patch", "output", |pass| linear_bf16_bias(
        context,
        pass,
        &px,
        &weights.patch_proj_w,
        &weights.patch_proj_b,
        &x,
        4
    ));
    stage!(x, "position", "output", |pass| patch_pos_embed_bf16(
        context,
        pass,
        &x,
        &weights.pos_embed,
        gh,
        gw
    ));
    stage!(h, "norm1", "output", |pass| layernorm_vision_bf16(
        context,
        pass,
        &x,
        &block.norm1_w,
        &block.norm1_b,
        &h,
        LAYERNORM_EPS
    ));
    stage!(qkv, "qkv", "output", |pass| linear_bf16_bias(
        context,
        pass,
        &h,
        &block.qkv_w,
        &block.qkv_b,
        &qkv,
        32
    ));
    stage!(qkv, "rotary", "output", |pass| if let Some((cos, sin)) = &rotary {
        crate::kernels::vision::qkv_rope_coefficients_bf16(
            context, pass, &qkv, cos, sin, gh, gw,
        )
    } else {
        qkv_rope_2d_bf16(context, pass, &qkv, vision.num_heads, gh, gw, ROPE_THETA)
    });
    stage!(attn, "attention", "output", |pass| attention_full_bf16(
        context,
        pass,
        &qkv,
        &attn,
        vision.num_heads,
        (vision.head_dim() as f32).powf(-0.5)
    ));
    stage!(tmp, "projection", "output", |pass| linear_bf16_bias(
        context,
        pass,
        &attn,
        &block.proj_w,
        &block.proj_b,
        &tmp,
        32
    ));
    stage!(x, "norm2", "input", |pass| add_bf16(context, pass, &x, &tmp, &x));
    stage!(h, "norm2", "output", |pass| layernorm_vision_bf16(
        context,
        pass,
        &x,
        &block.norm2_w,
        &block.norm2_b,
        &h,
        LAYERNORM_EPS
    ));
    stage!(mlp, "fc1", "output", |pass| linear_bf16_bias(
        context,
        pass,
        &h,
        &block.fc1_w,
        &block.fc1_b,
        &mlp,
        32
    ));
    stage!(mlp, "gelu_tanh", "output", |pass| gelu_vision_bf16(
        context,
        pass,
        &mlp,
        Gelu::Tanh
    ));
    stage!(tmp, "fc2", "output", |pass| linear_bf16_bias(
        context,
        pass,
        &mlp,
        &block.fc2_w,
        &block.fc2_b,
        &tmp,
        32
    ));
    stage!(x, "merger_norm", "input", |pass| add_bf16(context, pass, &x, &tmp, &x));
    stage!(h, "merger_norm", "output", |pass| layernorm_vision_bf16(
        context,
        pass,
        &x,
        &weights.merger_norm_w,
        &weights.merger_norm_b,
        &h,
        LAYERNORM_EPS
    ));
    stage!(mlp4, "merger_fc1", "output", |pass| linear_bf16_bias(
        context,
        pass,
        &h4,
        &weights.merger_fc1_w,
        &weights.merger_fc1_b,
        &mlp4,
        32
    ));
    stage!(mlp4, "gelu_erf", "output", |pass| gelu_vision_bf16(
        context,
        pass,
        &mlp4,
        Gelu::Erf
    ));
    stage!(merged, "merger_fc2", "output", |pass| linear_bf16_bias(
        context,
        pass,
        &mlp4,
        &weights.merger_fc2_w,
        &weights.merger_fc2_b,
        &merged,
        32
    ));
    println!(
        "{}",
        json!({"private_retained_sequence_metrics":{"patches":n,"stages":metrics}})
    );
    Some(merged.to_f32().unwrap())
}

fn full_norm_metrics(
    context: &MetalContext,
    weights: &super::super::vision_weights::VisionWeights,
    folder: &Path,
    case: &Value,
) {
    let Some(records) = case["full_states"].as_array() else { return };
    assert_eq!(records.len(), 27);
    let mut metrics = Vec::new();
    for (index, record) in records.iter().enumerate() {
        assert_eq!(record["block_index"], index);
        let block = &weights.blocks[index];
        for (norm_index, (name, weight, bias)) in [
            ("norm1", &block.norm1_w, &block.norm1_b),
            ("norm2", &block.norm2_w, &block.norm2_b),
        ]
        .iter()
        .enumerate()
        {
            let input_record = &record[format!("{name}_input")];
            let expected = checked_tensor(folder, &record[format!("{name}_output")]);
            let source = checked_tensor(folder, input_record);
            let shape = input_record["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
                .collect::<Vec<_>>();
            let input = Tensor::from_bytes(
                context,
                bytemuck::cast_slice(&source),
                &shape,
                DType::BF16,
            )
            .unwrap();
            let output = Tensor::zeros(context, &shape, DType::BF16).unwrap();
            let pass = context.begin().unwrap();
            layernorm_bf16(
                context,
                &pass,
                &input,
                weight,
                bias,
                &output,
                LAYERNORM_EPS,
            )
            .unwrap_or_else(|_| panic!("private full norm execution failed"));
            pass.commit_wait()
                .unwrap_or_else(|_| panic!("private full norm synchronization failed"));
            let actual = output.to_f32().unwrap();
            let (rms, cosine) = numerical_metric(&actual, &expected);
            let mismatches = actual
                .iter()
                .zip(&expected)
                .filter(|(a, b)| bf16::from_f32(**a).to_bits() != b.to_bits())
                .count();
            metrics.push(json!({"block_index":index,"norm_index":norm_index,"elements":actual.len(),"relative_rms":rms,"cosine":cosine,"mismatch_count":mismatches}));
        }
    }
    println!("{}", json!({"private_retained_full_norm_metrics":{"stages":metrics}}));
}

/// Each operator gets the source's input, so accumulated error from earlier
/// operators cannot hide the first arithmetic difference. These are private
/// fixture controls over existing kernels; the production graph is unchanged.
fn operator_metrics(
    context: &MetalContext,
    vision: &super::super::config::VisionConfig,
    weights: &super::super::vision_weights::VisionWeights,
    folder: &Path,
    case: &Value,
    gh: usize,
    gw: usize,
) -> Vec<Value> {
    let Some(operators) = case["operators"].as_array() else { return Vec::new() };
    let block = &weights.blocks[0];
    operators.iter().enumerate().map(|(index, record)| {
        let name = record["operator"].as_str().expect("private operator name missing");
        let source = checked_tensor(folder, &record["input"]);
        let expected = checked_tensor(folder, &record["output"]);
        let shape = |key: &str| record[key]["shape"].as_array().unwrap().iter()
            .map(|value| usize::try_from(value.as_u64().unwrap()).unwrap()).collect::<Vec<_>>();
        let input = Tensor::from_bytes(context, bytemuck::cast_slice(&source), &shape("input"), DType::BF16).unwrap();
        let output = Tensor::zeros(context, &shape("output"), DType::BF16).unwrap();
        let pass = context.begin().unwrap();
        let linear = match name {
            "patch" => Some((&weights.patch_proj_w, &weights.patch_proj_b)),
            "qkv" => Some((&block.qkv_w, &block.qkv_b)),
            "projection" => Some((&block.proj_w, &block.proj_b)),
            "fc1" => Some((&block.fc1_w, &block.fc1_b)),
            "fc2" => Some((&block.fc2_w, &block.fc2_b)),
            "merger_fc1" => Some((&weights.merger_fc1_w, &weights.merger_fc1_b)),
            "merger_fc2" => Some((&weights.merger_fc2_w, &weights.merger_fc2_b)),
            _ => None,
        };
        let result = if let Some((weight, bias)) = linear {
            gemm_bf16_nt_bias(context, &pass, &input, weight, bias, &output)
        } else {
            match name {
                "norm1" => layernorm_bf16(context, &pass, &input, &block.norm1_w, &block.norm1_b, &output, LAYERNORM_EPS),
                "norm2" => layernorm_bf16(context, &pass, &input, &block.norm2_w, &block.norm2_b, &output, LAYERNORM_EPS),
                "merger_norm" => layernorm_bf16(context, &pass, &input, &weights.merger_norm_w, &weights.merger_norm_b, &output, LAYERNORM_EPS),
                "attention" => attention_full_bf16(context, &pass, &input, &output, vision.num_heads, (vision.head_dim() as f32).powf(-0.5)),
                "position" | "rotary" | "gelu_tanh" | "gelu_erf" => {
                    // These kernels operate in place; preserve the exact fixture input.
                    output.write_bytes(bytemuck::cast_slice(&source)).unwrap();
                    match name {
                        "position" => patch_pos_embed_bf16(context, &pass, &output, &weights.pos_embed, gh, gw),
                        "rotary" => qkv_rope_2d_bf16(context, &pass, &output, vision.num_heads, gh, gw, ROPE_THETA),
                        "gelu_tanh" => gelu_bf16(context, &pass, &output, Gelu::Tanh),
                        _ => gelu_bf16(context, &pass, &output, Gelu::Erf),
                    }
                },
                _ => panic!("private operator unsupported"),
            }
        };
        result.unwrap_or_else(|_| panic!("private operator execution failed"));
        pass.commit_wait().unwrap_or_else(|_| panic!("private operator synchronization failed"));
        let actual = output.to_f32().unwrap_or_else(|_| panic!("private operator readback failed"));
        let (rms, cosine) = numerical_metric(&actual, &expected);
        if let Some((weight, bias)) = linear {
            let reference = Tensor::zeros(context, output.shape(), DType::BF16).unwrap();
            let pass = context.begin().unwrap();
            let streams = if name == "patch" { 4 } else { 32 };
            crate::kernels::vision::contraction_reference_bf16_bias(context, &pass, &input, weight, bias, &reference, streams)
                .unwrap_or_else(|_| panic!("private contraction reference execution failed"));
            pass.commit_wait().unwrap_or_else(|_| panic!("private contraction reference synchronization failed"));
            let values = reference.to_f32().unwrap();
            let (reference_rms, reference_cosine) = numerical_metric(&values, &expected);
            let mismatches = values.iter().zip(&expected).filter(|(a,b)| bf16::from_f32(**a).to_bits() != b.to_bits()).count();
            let ordinary_mismatches = actual.iter().zip(&expected).filter(|(a,b)| bf16::from_f32(**a).to_bits() != b.to_bits()).count();
            let production = Tensor::zeros(context, output.shape(), DType::BF16).unwrap();
            let pass = context.begin().unwrap();
            linear_bf16_bias(context, &pass, &input, weight, bias, &production, streams)
                .unwrap_or_else(|_| panic!("private production contraction execution failed"));
            pass.commit_wait().unwrap();
            let production = production.to_f32().unwrap();
            let production_mismatches = production.iter().zip(&expected).filter(|(a,b)| bf16::from_f32(**a).to_bits() != b.to_bits()).count();
            println!("{}", json!({"private_retained_contraction_metrics":{"patches":gh * gw,"operator_index":index,"streams":streams,"elements":values.len(),"relative_rms":reference_rms,"cosine":reference_cosine,"mismatch_count":mismatches,"ordinary_mismatch_count":ordinary_mismatches,"production_mismatch_count":production_mismatches}}));
            assert_eq!(production_mismatches, 0, "private vision contraction exceeds exact source gate");
        }
        let norm = match name {
            "norm1" => Some((&block.norm1_w, &block.norm1_b)),
            "norm2" => Some((&block.norm2_w, &block.norm2_b)),
            "merger_norm" => Some((&weights.merger_norm_w, &weights.merger_norm_b)),
            _ => None,
        };
        if norm.is_some() || matches!(name, "gelu_tanh" | "gelu_erf") {
            let reference = Tensor::zeros(context, output.shape(), DType::BF16).unwrap();
            let pass = context.begin().unwrap();
            let result = if let Some((weight, bias)) = norm {
                crate::kernels::vision::layernorm_reference_bf16(context, &pass, &input, weight, bias, &reference, LAYERNORM_EPS)
            } else {
                reference.write_bytes(bytemuck::cast_slice(&source)).unwrap();
                crate::kernels::vision::gelu_reference_bf16(context, &pass, &reference, name == "gelu_tanh")
            };
            result.unwrap_or_else(|_| panic!("private nonlinear reference execution failed"));
            pass.commit_wait().unwrap_or_else(|_| panic!("private nonlinear reference synchronization failed"));
            let values = reference.to_f32().unwrap();
            let (reference_rms, reference_cosine) = numerical_metric(&values, &expected);
            let mismatches = values.iter().zip(&expected).filter(|(a,b)| bf16::from_f32(**a).to_bits() != b.to_bits()).count();
            let ordinary_mismatches = actual.iter().zip(&expected).filter(|(a,b)| bf16::from_f32(**a).to_bits() != b.to_bits()).count();
            let production_mismatches;
            if let Some((weight, bias)) = norm {
                let production = Tensor::zeros(context, output.shape(), DType::BF16).unwrap();
                let pass = context.begin().unwrap();
                layernorm_vision_bf16(context, &pass, &input, weight, bias, &production, LAYERNORM_EPS).unwrap();
                pass.commit_wait().unwrap();
                let production = production.to_f32().unwrap();
                production_mismatches = Some(production.iter().zip(&expected).filter(|(a,b)| bf16::from_f32(**a).to_bits() != b.to_bits()).count());
            } else {
                let production = Tensor::zeros(context, output.shape(), DType::BF16).unwrap();
                production.write_bytes(bytemuck::cast_slice(&source)).unwrap();
                let pass = context.begin().unwrap();
                gelu_vision_bf16(context, &pass, &production, if name == "gelu_tanh" { Gelu::Tanh } else { Gelu::Erf }).unwrap();
                pass.commit_wait().unwrap();
                let production = production.to_f32().unwrap();
                production_mismatches = Some(production.iter().zip(&expected).filter(|(a,b)| bf16::from_f32(**a).to_bits() != b.to_bits()).count());
            }
            println!("{}", json!({"private_retained_nonlinear_metrics":{"patches":gh * gw,"operator_index":index,"elements":values.len(),"relative_rms":reference_rms,"cosine":reference_cosine,"mismatch_count":mismatches,"ordinary_mismatch_count":ordinary_mismatches,"production_mismatch_count":production_mismatches}}));
            assert!(production_mismatches.is_none_or(|count| count == 0), "private vision norm exceeds exact source gate");
        }
        if name == "position" {
            let reference = Tensor::zeros(context, output.shape(), DType::BF16).unwrap();
            reference.write_bytes(bytemuck::cast_slice(&source)).unwrap();
            let pass = context.begin().unwrap();
            crate::kernels::vision::patch_pos_reference_bf16(context, &pass, &reference, &weights.pos_embed, gh, gw).unwrap();
            pass.commit_wait().unwrap();
            let values = reference.to_f32().unwrap();
            let (reference_rms, reference_cosine) = numerical_metric(&values, &expected);
            let mismatches = values.iter().zip(&expected).filter(|(a,b)| bf16::from_f32(**a).to_bits() != b.to_bits()).count();
            let ordinary_mismatches = actual.iter().zip(&expected).filter(|(a,b)| bf16::from_f32(**a).to_bits() != b.to_bits()).count();
            println!("{}", json!({"private_retained_position_metrics":{"patches":gh * gw,"elements":values.len(),"relative_rms":reference_rms,"cosine":reference_cosine,"mismatch_count":mismatches,"ordinary_mismatch_count":ordinary_mismatches,"production_mismatch_count":ordinary_mismatches}}));
            assert_eq!(mismatches, 0, "private position witness exceeds exact source gate");
            assert_eq!(ordinary_mismatches, 0, "private vision position exceeds exact source gate");
        }
        if name == "rotary" {
            let reference = Tensor::zeros(context, output.shape(), DType::BF16).unwrap();
            reference.write_bytes(bytemuck::cast_slice(&source)).unwrap();
            let pass = context.begin().unwrap();
            crate::kernels::vision::qkv_rope_reference_bf16(context, &pass, &reference, vision.num_heads, gh, gw, ROPE_THETA).unwrap();
            pass.commit_wait().unwrap();
            let values = reference.to_f32().unwrap();
            let (reference_rms, reference_cosine) = numerical_metric(&values, &expected);
            let mismatches = values.iter().zip(&expected).filter(|(a,b)| bf16::from_f32(**a).to_bits() != b.to_bits()).count();
            let ordinary_mismatches = actual.iter().zip(&expected).filter(|(a,b)| bf16::from_f32(**a).to_bits() != b.to_bits()).count();
            println!("{}", json!({"private_retained_rotary_metrics":{"patches":gh * gw,"elements":values.len(),"relative_rms":reference_rms,"cosine":reference_cosine,"mismatch_count":mismatches,"ordinary_mismatch_count":ordinary_mismatches}}));
            rotary_source_metrics(context, &source, &expected, &values, case, gh * gw, gw);
        }
        if name == "attention" && record["stages"].is_object() {
            attention_order_metrics(context, &input, folder, record, &expected, &actual, gh * gw);
            let n = input.shape()[0];
            let plane = vision.num_heads * n * n;
            let stages = Tensor::zeros(context, &[3, vision.num_heads, n, n], DType::F32).unwrap();
            let traced = Tensor::zeros(context, output.shape(), DType::BF16).unwrap();
            let pass = context.begin().unwrap();
            crate::kernels::vision::attention_stages_bf16(context, &pass, &input, &traced, &stages, vision.num_heads, (vision.head_dim() as f32).powf(-0.5))
                .unwrap_or_else(|_| panic!("private attention trace execution failed"));
            pass.commit_wait().unwrap_or_else(|_| panic!("private attention trace synchronization failed"));
            let values = stages.to_f32().unwrap();
            let stage_metrics = ["qk", "scaled", "probabilities"].iter().enumerate().map(|(stage_index, key)| {
                let expected = checked_tensor(folder, &record["stages"][key]);
                let (rms, cosine) = numerical_metric(&values[stage_index * plane..(stage_index + 1) * plane], &expected);
                let mismatches = values[stage_index * plane..(stage_index + 1) * plane].iter().zip(&expected).filter(|(a,b)| bf16::from_f32(**a).to_bits()!=b.to_bits()).count();
                json!({"stage_index":stage_index,"elements":plane,"relative_rms":rms,"cosine":cosine,"mismatch_count":mismatches})
            }).collect::<Vec<_>>();
            let traced = traced.to_f32().unwrap();
            let mismatches = traced.iter().zip(&actual).filter(|(a,b)| a.to_bits() != b.to_bits()).count();
            let ordinary = actual.iter().map(|value| bf16::from_f32(*value)).collect::<Vec<_>>();
            let (trace_rms, _) = numerical_metric(&traced, &ordinary);
            println!("{}", json!({"private_retained_attention_stage_metrics":{"patches":n,"stages":stage_metrics,"trace_to_ordinary_relative_rms":trace_rms,"trace_ordinary_mismatch_count":mismatches}}));
            assert_eq!(mismatches, 0, "private attention trace perturbs ordinary arithmetic");
            assert!(stage_metrics.iter().all(|metric| metric["mismatch_count"] == 0), "private production attention stages exceed exact source gate");
            let source_mismatches = actual.iter().zip(&expected).filter(|(a,b)| bf16::from_f32(**a).to_bits()!=b.to_bits()).count();
            assert_eq!(source_mismatches, 0, "private production attention exceeds exact source gate");
        }
        json!({"operator_index":index,"elements":actual.len(),"relative_rms":rms,"cosine":cosine})
    }).collect()
}

#[test]
#[ignore = "requires an explicitly selected retained artifact, source-bound fixture and GPU lease"]
fn retained_lily_tower_source_reference() {
    let artifact = std::path::PathBuf::from(
        std::env::var_os("STARK_LILY_MODEL_PATH")
            .expect("set private retained artifact path"),
    );
    let reference = std::path::PathBuf::from(
        std::env::var_os("STARK_LILY_VISION_REFERENCE_PATH")
            .expect("set private retained tower reference path"),
    );
    let manifest_bytes = fs::read(reference.join("manifest.json"))
        .expect("private tower manifest unavailable");
    assert!(manifest_bytes.len() <= 1024 * 1024);
    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .expect("private tower manifest invalid");
    assert_eq!(manifest["kind"], "RetainedLilyTowerSourceReference");
    assert_eq!(manifest["source_revision"], "2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397");
    assert_eq!(
        manifest["source_sha256"]["modeling_qwen4_exp.py"],
        "e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df"
    );
    assert_eq!(manifest["gate"], json!({"relative_rms_max":0.02,"cosine_min":0.999}));
    let contract_bytes = fs::read(artifact.join("stark-image-artifact.json"))
        .expect("private retained contract unavailable");
    assert_eq!(hex(&sha256(&contract_bytes)), manifest["artifact_manifest_sha256"]);
    let contract: Value = serde_json::from_slice(&contract_bytes)
        .expect("private retained contract invalid");
    assert_eq!(contract, manifest["artifact_contract"]);
    assert_eq!(
        contract["artifact_sha256"],
        "18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4"
    );
    for (file, key) in [
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "index_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
        ("preprocessor_config.json", "processor_sha256"),
        ("chat_template.jinja", "template_sha256"),
    ] {
        assert!(fs::metadata(artifact.join(file)).unwrap().len() <= 32 * 1024 * 1024);
        assert_eq!(
            hex(&sha256(&fs::read(artifact.join(file)).unwrap())),
            contract[key]
        );
    }
    let config_bytes = fs::read(artifact.join("config.json")).unwrap();
    let raw_config: Value = serde_json::from_slice(&config_bytes).unwrap();
    assert_eq!(manifest["vision_config"], raw_config["vision_config"]);
    let config = Qwen4ExpConfig::from_json(&config_bytes)
        .unwrap_or_else(|_| panic!("private tower config admission failed"));
    let vision = config.vision.as_ref().expect("private retained tower config missing");
    assert_eq!((vision.depth, vision.hidden_size, vision.num_heads), (27, 1152, 16));
    let context = MetalContext::new()
        .unwrap_or_else(|_| panic!("private tower Metal setup failed"));
    gelu_domain_metrics(&context);
    let checkpoint = Checkpoint::open(&artifact)
        .unwrap_or_else(|_| panic!("private tower header admission failed"));
    let loader =
        Loader::new(&context, checkpoint, config.quantization, &[], Box::new(|_| 4));
    let weights = super::super::vision_weights::load(&loader, vision)
        .unwrap_or_else(|_| panic!("private tower weights failed"));
    assert_eq!(weights.bytes(), 897_862_112);
    let hashes = &manifest["source_visual_tensor_sha256"];
    for (name, tensor) in [
        ("patch_embed.proj.weight", &weights.patch_proj_w),
        ("patch_embed.proj.bias", &weights.patch_proj_b),
        ("pos_embed.weight", &weights.pos_embed),
        ("merger.norm.weight", &weights.merger_norm_w),
        ("merger.norm.bias", &weights.merger_norm_b),
        ("merger.linear_fc1.weight", &weights.merger_fc1_w),
        ("merger.linear_fc1.bias", &weights.merger_fc1_b),
        ("merger.linear_fc2.weight", &weights.merger_fc2_w),
        ("merger.linear_fc2.bias", &weights.merger_fc2_b),
    ] {
        check_weight(hashes, name, tensor);
    }
    for (index, block) in weights.blocks.iter().enumerate() {
        for (name, tensor) in [
            ("norm1.weight", &block.norm1_w),
            ("norm1.bias", &block.norm1_b),
            ("attn.qkv.weight", &block.qkv_w),
            ("attn.qkv.bias", &block.qkv_b),
            ("attn.proj.weight", &block.proj_w),
            ("attn.proj.bias", &block.proj_b),
            ("norm2.weight", &block.norm2_w),
            ("norm2.bias", &block.norm2_b),
            ("mlp.linear_fc1.weight", &block.fc1_w),
            ("mlp.linear_fc1.bias", &block.fc1_b),
            ("mlp.linear_fc2.weight", &block.fc2_w),
            ("mlp.linear_fc2.bias", &block.fc2_b),
        ] {
            check_weight(hashes, &format!("blocks.{index}.{name}"), tensor);
        }
    }
    let mut scratch = None;
    let mut metrics = Vec::new();
    let mut operators = Vec::new();
    for (case_index, case) in
        manifest["cases"].as_array().expect("private cases missing").iter().enumerate()
    {
        let name = case["name"].as_str().unwrap();
        assert!(matches!(name, "square_256" | "stress_32"));
        let folder = reference.join(name);
        let grid = case["grid"].as_array().unwrap();
        assert_eq!(grid[0], 1);
        let gh = usize::try_from(grid[1].as_u64().unwrap()).unwrap();
        let gw = usize::try_from(grid[2].as_u64().unwrap()).unwrap();
        assert!(gh * gw <= 256);
        let rgb = checked_file(
            &folder,
            &json!({"file":"rgb.u8","sha256":case["rgb_sha256"]}),
            256 * 256 * 3,
        );
        assert_eq!(rgb.len(), gh * gw * 16 * 16 * 3);
        let pixels = checked_tensor(&folder, &case["pixels"]);
        assert_eq!(pixels.len(), gh * gw * vision.patch_dim());
        let pixels = pixels.into_iter().map(f32::from).collect::<Vec<_>>();
        operators.push(json!({"case_index":case_index,"metrics":operator_metrics(&context, vision, &weights, &folder, case, gh, gw)}));
        full_norm_metrics(&context, &weights, &folder, case);
        let sequence = first_block_metrics(
            &context,
            vision,
            &weights,
            &mut scratch,
            &folder,
            case,
            &pixels,
            gh,
            gw,
        );
        for stage in case["stages"].as_array().unwrap() {
            let depth = usize::try_from(stage["blocks"].as_u64().unwrap()).unwrap();
            assert!(matches!(depth, 0 | 1 | 27));
            let expected = checked_tensor(&folder, &stage["merged"]);
            let output = forward_with(
                &context,
                vision,
                &weights,
                &mut scratch,
                &pixels,
                gh,
                gw,
                depth,
            )
            .unwrap_or_else(|_| panic!("private retained tower execution failed"));
            let actual = output
                .merged
                .to_f32()
                .unwrap_or_else(|_| panic!("private retained tower readback failed"));
            let (rms, cosine) = numerical_metric(&actual, &expected);
            if depth == 1 {
                if let Some(sequence) = &sequence {
                    let mismatches = actual
                        .iter()
                        .zip(sequence)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    println!(
                        "{}",
                        json!({"private_retained_sequence_observer":{"case_index":case_index,"mismatch_count":mismatches}})
                    );
                    assert_eq!(
                        mismatches, 0,
                        "private sequence readbacks perturb ordinary graph"
                    );
                }
            }
            metrics.push(json!({"case_index":case_index,"blocks":depth,"elements":actual.len(),"relative_rms":rms,"cosine":cosine}));
        }
    }
    println!("{}", json!({"private_retained_tower_metrics":metrics}));
    println!("{}", json!({"private_retained_operator_metrics":operators}));
    // Independent same-input source control: this is tight enough to reject
    // the former unnormalized-BF16-P online policy (~0.4% RMS) while retaining
    // room for F32 reduction order and BF16 halfway rounding across devices.
    for (case, result) in manifest["cases"].as_array().unwrap().iter().zip(&operators) {
        if let Some(records) = case["operators"].as_array() {
            for (record, metric) in
                records.iter().zip(result["metrics"].as_array().unwrap())
            {
                if record["operator"] == "attention" {
                    assert!(
                        metric["relative_rms"].as_f64().unwrap() <= 0.001
                            && metric["cosine"].as_f64().unwrap() >= 0.999999,
                        "private retained attention exceeds same-input source gate"
                    );
                }
            }
        }
    }
    assert!(
        metrics.iter().all(|metric| metric["relative_rms"].as_f64().unwrap() <= 0.02
            && metric["cosine"].as_f64().unwrap() >= 0.999),
        "private retained tower exceeds frozen source gate"
    );
}
