use stark_engine_macros::stark_test;

use super::{KVCacheState, checked_cache_capacity, checked_preparation_rows};

#[stark_test]
fn cache_capacity_and_preparation_reject_overflow_before_dispatch() {
    assert!(checked_cache_capacity(u32::MAX, 64, 2).is_err());
    assert!(checked_cache_capacity(u32::MAX - 511, 64, 2).is_err());
    assert!(checked_cache_capacity(u32::MAX - 1024, u32::MAX, 8).is_err());
    assert_eq!(checked_cache_capacity(1024, 64, 2).unwrap(), (2048, 262144));
    assert!(checked_preparation_rows(u32::MAX, 1, u32::MAX).is_err());
    assert!(checked_preparation_rows(2048, 1, 2048).is_err());
    assert!(checked_preparation_rows(0, 1025, 2048).is_err());
    assert_eq!(checked_preparation_rows(1024, 1024, 2048).unwrap(), 2048);
}

fn accept(
    state: &mut KVCacheState,
    indices: &[u32],
) -> Vec<(u32, u32)> {
    state.accept(indices).into_iter().map(|copy| (copy.source, copy.destination)).collect()
}

#[stark_test]
fn ring_accepts_append_then_wrap() {
    let mut state = KVCacheState::ring(2);

    assert_eq!(accept(&mut state, &[1]), [(1, 0)]);
    assert_eq!(state.view().prefix_len(), 1);

    assert!(accept(&mut state, &[0]).is_empty());
    assert_eq!(state.view().prefix_len(), 2);

    assert_eq!(accept(&mut state, &[0]), [(2, 0)]);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 1);
    assert_eq!(state.required_prefix_len(99), 2);
}

#[stark_test]
fn ring_accept_over_capacity_keeps_last() {
    let mut state = KVCacheState::ring(2);

    // Only the last `capacity` tokens survive; earlier copies are dead.
    assert_eq!(accept(&mut state, &[0, 1, 2]), [(2, 0)]);
    assert_eq!(state.view().prefix_len(), 2);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 1);

    // Same with a wrapped ring: destinations keep rotating.
    assert_eq!(accept(&mut state, &[0, 1, 2]), [(3, 0), (4, 1)]);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 0);
}

#[stark_test]
fn source_ring_origin_tracks_bulk_temporal_reset_and_survives_single_updates() {
    // MLX-LM RotatingKVCache's bulk update is temporal; the next M=1
    // update trims to the window and overwrites slot zero. Later M=1 updates
    // keep that physical origin, even while the native ring offset advances.
    let mut state = KVCacheState::ring(512);
    accept(&mut state, &(0..516).collect::<Vec<_>>());
    assert_eq!(state.view().source_ring_origin(), 4);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 4);
    accept(&mut state, &[0]);
    accept(&mut state, &[0]);
    assert_eq!(state.view().source_ring_origin(), 4);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 6);
    accept(&mut state, &[0, 1, 2]);
    assert_eq!(state.view().source_ring_origin(), 9);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 9);

    let mut fresh = KVCacheState::ring(512);
    accept(&mut fresh, &(0..511).collect::<Vec<_>>());
    assert_eq!(fresh.view().source_ring_origin(), 0);
    accept(&mut fresh, &[0]);
    accept(&mut fresh, &[0]);
    assert_eq!(fresh.view().source_ring_origin(), 0);
    assert_eq!(fresh.view().ring_params().unwrap().ring_offset, 1);
    assert_eq!(KVCacheState::ring(512).view().source_ring_origin(), 0);
}
