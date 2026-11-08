//! Checked request allocation inventories for hybrid-family admission.

pub(super) fn bytes(
    shape: &[usize],
    element_bytes: usize,
) -> Option<usize> {
    shape.iter().try_fold(element_bytes, |total, &dimension| total.checked_mul(dimension))
}

pub(super) fn sum(parts: &[usize]) -> Option<usize> {
    parts.iter().try_fold(0usize, |total, &part| total.checked_add(part))
}

#[cfg(feature = "metal")]
pub(super) fn gpu_buffer_fits(
    shape: &[usize],
    element_bytes: usize,
) -> Option<()> {
    gpu_buffer_fits_limit(shape, element_bytes, super::weight_metal::max_buffer_length().ok()?)
}

#[cfg(any(feature = "metal", test))]
fn gpu_buffer_fits_limit(
    shape: &[usize],
    element_bytes: usize,
    device_limit: usize,
) -> Option<()> {
    let elements = bytes(shape, 1)?;
    let required = elements.checked_mul(element_bytes)?;
    // Hybrid shaders receive 32-bit element offsets and geometry fields.
    (elements <= u32::MAX as usize && required <= device_limit).then_some(())
}

pub(super) fn growth_capacity(
    requested: u32,
    maximum: usize,
    minimum: usize,
) -> Option<usize> {
    let requested = requested as usize;
    if requested == 0 || requested > maximum {
        return None;
    }
    Some(requested.checked_next_power_of_two()?.max(minimum).min(maximum))
}

pub(super) fn host_workspace(
    context: u32,
    input: u32,
    vocab: usize,
) -> Option<usize> {
    // History/output vectors, prompt copies, readout and exact sampler arrays.
    sum(&[bytes(&[context as usize], 32)?, bytes(&[input as usize], 16)?, bytes(&[vocab], 64)?, 8 * 1024 * 1024])
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn growth_admits_real_rounding_boundaries_and_checks_overflow() {
        assert_eq!(growth_capacity(65, 1000, 64), Some(128));
        assert_eq!(growth_capacity(999, 1000, 64), Some(1000));
        assert_eq!(growth_capacity(1, 32, 64), Some(32));
        assert_eq!(growth_capacity(1001, 1000, 64), None);
        assert!(bytes(&[usize::MAX, 2], 4).is_none());
        assert!(sum(&[usize::MAX, 1]).is_none());
    }

    #[stark_test]
    fn gpu_admission_checks_device_bytes_and_shader_element_offsets() {
        let limit = u32::MAX as usize / (8 * 128);
        assert!(gpu_buffer_fits_limit(&[limit, 8, 128], 4, usize::MAX).is_some());
        assert!(gpu_buffer_fits_limit(&[limit + 1, 8, 128], 4, usize::MAX).is_none());
        assert!(gpu_buffer_fits_limit(&[64, 512], 4, 128 * 1024).is_some());
        assert!(gpu_buffer_fits_limit(&[65, 512], 4, 128 * 1024).is_none());
    }
}
