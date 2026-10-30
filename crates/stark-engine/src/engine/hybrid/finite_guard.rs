//! Finite-value checks for native hybrid readouts.

/// Returns whether every value is finite, including for an empty slice.
/// Callers must check their required shape and length separately.
#[inline]
pub fn all_finite(values: &[f32]) -> bool {
    // The fixed-size, non-short-circuit reduction lets LLVM vectorize each
    // chunk. Keep early rejection between chunks and check the scalar tail.
    let (chunks, remainder) = values.as_chunks::<32>();
    !chunks
        .iter()
        .any(|chunk| chunk.iter().fold(false, |contains_nonfinite, value| contains_nonfinite | !value.is_finite()))
        && remainder.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::all_finite;

    fn finite_fixture(count: usize) -> Vec<f32> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let bits = state as u32;
                f32::from_bits(if bits & 0x7f80_0000 == 0x7f80_0000 {
                    bits ^ 0x0080_0000
                } else {
                    bits
                })
            })
            .collect()
    }

    #[stark_test]
    fn accepts_empty_finite_ieee_values_and_chunk_tails() {
        assert!(all_finite(&[]));
        let mut values = vec![0.0, -0.0, f32::MAX, -f32::MAX, f32::from_bits(1), -f32::from_bits(1)];
        for exponent in 0..255 {
            for fraction in [0, 1, 0x003f_ffff, 0x007f_ffff] {
                let bits = (exponent << 23) | fraction;
                values.push(f32::from_bits(bits));
                values.push(f32::from_bits(bits | 0x8000_0000));
            }
        }
        values.extend(finite_fixture(248_320));
        for len in (0..=67).chain([values.len()]) {
            assert!(all_finite(&values[..len]));
        }
    }

    #[stark_test]
    fn rejects_nonfinite_ieee_values_at_chunk_boundaries_and_tails() {
        for len in (1..=67).chain([248_320]) {
            let finite = finite_fixture(len);
            for position in [0, len / 2, len - 1, 31.min(len - 1), 32.min(len - 1)] {
                for bits in [0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xff80_0001, 0x7fc0_0000, 0xffff_ffff] {
                    let mut values = finite.clone();
                    values[position] = f32::from_bits(bits);
                    assert!(!all_finite(&values));
                    assert_eq!(all_finite(&values), values.iter().all(|value| value.is_finite()));
                }
            }
        }
    }
}
