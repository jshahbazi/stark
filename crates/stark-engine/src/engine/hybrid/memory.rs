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

/// Additional peak for geometrically grown owners. Current owners already
/// reduce available RAM; the largest old generation coexists with its new
/// replacement until the copying command completes.
pub(super) fn growing_owner_bytes(
    current: usize,
    target: usize,
    minimum: usize,
    mut allocation: impl FnMut(usize) -> Option<usize>,
) -> Option<usize> {
    if target <= current {
        return Some(0);
    }
    let current_bytes = if current == 0 {
        0
    } else {
        allocation(current)?
    };
    let predecessor = if target <= minimum {
        0
    } else {
        target.checked_next_power_of_two()?.checked_div(2)?.max(current)
    };
    let overlap = if predecessor == 0 {
        0
    } else {
        allocation(predecessor)?
    };
    allocation(target)?.checked_sub(current_bytes)?.checked_add(overlap)
}

/// Peak for repeated pushes into retained Vec/VecDeque descriptor owners.
pub(super) fn vector_growth_bytes<T>(
    capacity: usize,
    needed: usize,
) -> Option<usize> {
    if needed <= capacity {
        Some(0)
    } else {
        let minimum = if size_of::<T>() == 1 {
            8
        } else if size_of::<T>() <= 1024 {
            4
        } else {
            1
        };
        let mut target = capacity;
        let mut previous = capacity;
        while target < needed {
            previous = target;
            target = target.checked_mul(2)?.max(minimum);
        }
        bytes(&[target.checked_sub(capacity)?.checked_add(previous)?], size_of::<T>())
    }
}

/// One bulk resize uses a single amortized replacement allocation.
pub(super) fn vector_resize_bytes<T>(
    capacity: usize,
    needed: usize,
) -> Option<usize> {
    if needed <= capacity {
        Some(0)
    } else {
        bytes(&[needed.max(capacity.checked_mul(2)?).max(4)], size_of::<T>())
    }
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

#[cfg(test)]
pub(super) fn host_workspace(
    context: u32,
    input: u32,
    vocab: usize,
) -> Option<usize> {
    host_workspace_for_request(context, input, vocab, false)
}

pub(super) fn host_workspace_for_request(
    context: u32,
    input: u32,
    vocab: usize,
    greedy_without_mtp: bool,
) -> Option<usize> {
    // History/output vectors and prompt/text copies are fresh request owners.
    // Greedy hybrid readout is F32 throughout Model::step, mapped_many and
    // PendingDecodeGraph::wait. At scalar readback, the previous host row,
    // new device row and new host row can coexist: three F32 rows. Greedy
    // sampling scans logits without full-vocabulary F64 probability arrays.
    // Unknown/stochastic/MTP paths retain the existing exact-sampler bound;
    // their independent arrays cannot credit dormant worker sampler capacity.
    let readout_and_sampling = if greedy_without_mtp {
        3 * size_of::<f32>()
    } else {
        64
    };
    sum(&[
        bytes(&[context as usize], 32)?,
        bytes(&[input as usize], 16)?,
        bytes(&[vocab], readout_and_sampling)?,
        8 * 1024 * 1024,
    ])
}

pub(super) fn scoring_host_workspace(
    vocab: usize,
    additional_device_readout: bool,
) -> Option<usize> {
    // One F32 row is returned. Scalar Metal matvec also allocates a new
    // device readout; graph/prepaid device outputs are already inventoried.
    // Reduction streams scalars and keeps at most sixteen candidates. Parsed
    // input is already resident before admission; no generation history or
    // stochastic sampler arrays are allocated by exact scoring.
    sum(&[bytes(&[vocab, 1 + usize::from(additional_device_readout)], size_of::<f32>())?, 64 * 1024])
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

    #[stark_test]
    fn retained_owners_are_not_rebilled_and_growth_keeps_overlap() {
        let allocation = |rows| bytes(&[rows], 4);
        assert_eq!(growing_owner_bytes(128, 64, 64, allocation), Some(0));
        assert_eq!(growing_owner_bytes(128, 128, 64, allocation), Some(0));
        assert_eq!(growing_owner_bytes(0, 64, 64, allocation), Some(256));
        assert_eq!(growing_owner_bytes(64, 128, 64, allocation), Some(512));
        assert_eq!(growing_owner_bytes(64, 1000, 64, allocation), Some((1000 - 64 + 512) * 4));
        assert_eq!(growing_owner_bytes(0, usize::MAX, 64, allocation), None);
        assert_eq!(vector_growth_bytes::<u32>(1, 2), Some(4 * 4));
        assert_eq!(vector_growth_bytes::<u32>(2, 3), Some(4 * 4));
        assert_eq!(vector_growth_bytes::<u32>(3, 4), Some(6 * 4));
        assert_eq!(vector_growth_bytes::<u32>(3, 7), Some((12 - 3 + 6) * 4));
        assert_eq!(vector_growth_bytes::<u32>(1024, 1033), Some(2048 * 4));
        assert_eq!(vector_growth_bytes::<u8>(0, 1), Some(8));
        assert_eq!(vector_growth_bytes::<u32>(128, 128), Some(0));
        assert_eq!(vector_growth_bytes::<u32>(128, 129), Some(1024));
        assert_eq!(vector_growth_bytes::<u32>(0, 16), Some((16 + 8) * 4));
        assert_eq!(vector_growth_bytes::<u32>(6, 50), Some((96 - 6 + 48) * 4));
        assert_eq!(vector_resize_bytes::<u32>(0, 1000), Some(4000));
        assert_eq!(vector_resize_bytes::<u32>(1000, 999), Some(0));
    }

    #[stark_test]
    fn greedy_host_workspace_omits_absent_full_vocabulary_sampler_arrays() {
        let cold = host_workspace(4096, 1025, 1000).unwrap();
        let greedy = host_workspace_for_request(4096, 1025, 1000, true).unwrap();
        assert_eq!(cold - greedy, (64 - 3 * size_of::<f32>()) * 1000);
        assert_eq!(host_workspace_for_request(4096, 1025, 1000, false), Some(cold));
        assert_eq!(host_workspace_for_request(4096, 1025, usize::MAX, true), None);
    }

    #[stark_test]
    fn exact_scoring_host_workspace_has_no_generation_sampler_or_output_reservation() {
        assert_eq!(scoring_host_workspace(1000, false), Some(4000 + 64 * 1024));
        assert_eq!(scoring_host_workspace(1000, true), Some(8000 + 64 * 1024));
        assert!(scoring_host_workspace(1000, true).unwrap() < host_workspace(4096, 4096, 1000).unwrap());
        assert_eq!(scoring_host_workspace(usize::MAX, false), None);
    }
}
