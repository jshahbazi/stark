//! Conservative request-owned tower allocation accounting.
//!
//! Count submitted-command storage without assuming that scratch dropped by a
//! Rust scope is reusable before GPU completion. Weights are already resident
//! at request admission; these bytes cover patches, uploaded constants, BF16
//! boundaries, F32 reductions, projected readbacks and allocator slack.

pub(super) fn tower_workspace_bytes(
    rows: u32,
    patch_width: u32,
    hidden: u32,
    intermediate: u32,
    output: u32,
    live_blocks: u32,
    eager_scores: bool,
) -> Option<usize> {
    if rows == 0 || hidden == 0 || patch_width == 0 {
        return None;
    }
    let n = rows as usize;
    let h = hidden as usize;
    let i = intermediate as usize;
    // Across all currently admitted towers, a block allocates fewer than
    // 128 bytes per hidden coordinate and 16 per intermediate coordinate:
    // this includes Q/K/V padding, clipped linears, norms/residuals, full F32
    // linear reductions and BF16 outputs. Eager QK/probabilities reuse one
    // F32+BF16 score matrix across heads (6 bytes per pair).
    let linear = n.checked_mul(h)?.checked_mul(128)?.checked_add(n.checked_mul(i)?.checked_mul(16)?)?;
    let scores = if eager_scores {
        n.checked_mul(n)?.checked_mul(6)?
    } else {
        0
    };
    let blocks = linear.checked_add(scores)?.checked_add(4 * 1024 * 1024)?.checked_mul(live_blocks as usize)?;
    // CPU patches + uploaded patch constants, learned/rotary positions,
    // initial projection, merger/projector reductions and both readback copies.
    let edges = n
        .checked_mul(patch_width as usize)?
        .checked_mul(8)?
        .checked_add(n.checked_mul(h)?.checked_mul(64)?)?
        .checked_add(n.checked_mul(output as usize)?.checked_mul(16)?)?;
    blocks.checked_add(edges)?.checked_add(32 * 1024 * 1024)
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn tower_budget_includes_quadratic_scores_submission_lifetimes_and_overflow() {
        let eager = tower_workspace_bytes(4096, 1536, 1152, 4304, 4096, 27, true).unwrap();
        let online = tower_workspace_bytes(4096, 1536, 1152, 4304, 4096, 27, false).unwrap();
        assert_eq!(eager - online, 27 * 4096 * 4096 * 6);
        let serial = tower_workspace_bytes(4096, 588, 1152, 4304, 4096, 1, true).unwrap();
        assert!(serial < eager);
        assert!(serial > 4096 * 4096 * 6 + 4096 * 4304 * 6);
        assert!(tower_workspace_bytes(u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, true).is_none());
        assert!(tower_workspace_bytes(0, 1536, 1152, 4304, 4096, 27, true).is_none());
    }
}
