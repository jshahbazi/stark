//! Scalar control operations shared by both native hybrid forward paths.

use super::{Error, Result};

pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

pub fn rms_norm(
    x: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    if x.is_empty()
        || x.len() != weight.len()
        || !eps.is_finite()
        || eps <= 0.0
        || x.iter().chain(weight).any(|value| !value.is_finite())
    {
        return Err(Error::Invalid("RMS normalization dimensions or values"));
    }
    let mean = x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
    let scale = (mean + eps).sqrt().recip();
    Ok(x.iter().zip(weight).map(|(&value, &w)| value * scale * w).collect())
}

pub fn softmax(x: &[f32]) -> Result<Vec<f32>> {
    if x.is_empty() || x.iter().any(|value| value.is_nan() || *value == f32::INFINITY) {
        return Err(Error::Invalid("softmax input"));
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Err(Error::Invalid("all softmax positions are masked"));
    }
    let mut result: Vec<f32> = x.iter().map(|value| (value - max).exp()).collect();
    let sum = result.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::Invalid("softmax normalization"));
    }
    result.iter_mut().for_each(|value| *value /= sum);
    Ok(result)
}

/// Stable expert/block selection. Correction biases affect selection only;
/// callers retain the original router probabilities for the mixture weights.
pub fn topk(
    x: &[f32],
    k: usize,
) -> Result<Vec<usize>> {
    if k == 0 || k > x.len() || x.iter().any(|value| value.is_nan()) {
        return Err(Error::Invalid("top-k input"));
    }
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.sort_unstable_by(|&a, &b| x[b].total_cmp(&x[a]).then_with(|| a.cmp(&b)));
    indices.truncate(k);
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn stable_softmax_and_masking() {
        let p = softmax(&[10_000.0, 10_000.0, f32::NEG_INFINITY]).unwrap();
        assert_eq!(p, [0.5, 0.5, 0.0]);
        assert!(softmax(&[f32::NEG_INFINITY]).is_err());
        assert!(softmax(&[f32::NAN]).is_err());
    }

    #[stark_test]
    fn selection_bias_does_not_change_original_scores() {
        let scores = [0.25, 0.75, 0.5];
        let selected = topk(&[scores[0] + 1.0, scores[1], scores[2]], 2).unwrap();
        assert_eq!(selected, [0, 1]);
        assert_eq!(scores[selected[0]] / (scores[0] + scores[1]), 0.25);
        assert_eq!(topk(&[1.0, 1.0, 1.0], 2).unwrap(), [0, 1]);
    }

    #[stark_test]
    fn rms_normalization_and_extreme_gates() {
        let normalized = rms_norm(&[3.0, 4.0], &[1.0, 2.0], 1e-6).unwrap();
        assert!((normalized[0] - 3.0 / 12.5_f32.sqrt()).abs() < 1e-6);
        assert!((normalized[1] - 8.0 / 12.5_f32.sqrt()).abs() < 1e-6);
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert_eq!(sigmoid(1000.0), 1.0);
        assert!(rms_norm(&[1.0], &[], 1e-6).is_err());
    }
}
