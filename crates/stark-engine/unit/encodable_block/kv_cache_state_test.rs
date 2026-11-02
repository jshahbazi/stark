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
fn full_cache_rewinds_to_shorter_prefix_only() {
    let mut state = KVCacheState::full();
    assert!(accept(&mut state, &(0..8).collect::<Vec<_>>()).is_empty());
    assert_eq!(state.view().prefix_len(), 8);

    assert!(!state.can_rewind(9));
    assert!(!state.rewind(9));
    assert_eq!(state.view().prefix_len(), 8);

    assert!(state.can_rewind(4));
    assert!(state.rewind(4));
    assert_eq!(state.view().prefix_len(), 4);
    // The next batch is written immediately after the kept prefix.
    assert!(accept(&mut state, &[0, 1]).is_empty());
    assert_eq!(state.view().prefix_len(), 6);

    assert!(state.rewind(0));
    assert_eq!(state.view().prefix_len(), 0);
}

#[stark_test]
fn ring_cache_never_rewinds() {
    let mut state = KVCacheState::ring(4);
    accept(&mut state, &[0, 1, 2]);
    for length in [0, 1, 3] {
        assert!(!state.can_rewind(length));
        assert!(!state.rewind(length));
    }
    assert_eq!(state.view().prefix_len(), 3);
}
