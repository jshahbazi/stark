// SPDX-License-Identifier: Apache-2.0
//! Exact greedy reduction, finite obligations and submitted-owner readback.

use stark_engine_macros::stark_test;

use super::*;

fn first_index(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

#[stark_test]
fn gpu_greedy_matches_first_index_ties_signed_zero_negatives_and_tails() {
    let mut cases =
        vec![vec![-0.0, 0.0], vec![0.0, -0.0], vec![-5.0, -2.0, -2.0, -3.0], vec![-f32::MAX], vec![f32::MAX, f32::MAX]];
    for bits in [1u32, 2, 0x007f_ffff, 0x0080_0000, 0x3f7f_ffff, 0x3f80_0000, 0x3f80_0001, 0x7f7f_fffe] {
        let a = f32::from_bits(bits);
        let b = f32::from_bits(bits + 1);
        cases.extend([vec![0.0, a], vec![-a, -b], vec![a, b, a], vec![-b, -a, -b]]);
    }
    cases.extend([
        vec![f32::from_bits(0x8000_0001), -0.0],
        vec![-0.0, f32::from_bits(0x8000_0001)],
        vec![f32::from_bits(1), 0.0, f32::from_bits(1)],
    ]);
    let mut subnormal_tiles = vec![f32::from_bits(0x8000_0002); 2051];
    subnormal_tiles[1024] = f32::from_bits(0x8000_0001);
    subnormal_tiles[2050] = f32::from_bits(0x8000_0001);
    cases.push(subnormal_tiles.clone());
    subnormal_tiles[1024] = f32::from_bits(1);
    subnormal_tiles[2050] = f32::from_bits(2);
    cases.push(subnormal_tiles);
    for len in [1, 13, 31, 32, 255, 256, 257, 1023, 1024, 1025, 154_880] {
        let mut values = (0..len).map(|i| -(i as f32) - 1.0).collect::<Vec<_>>();
        values[len - 1] = 3.0;
        cases.push(values.clone());
        values[(len - 1).min(257)] = 3.0;
        if len > 1024 {
            values[1024] = 3.0;
        }
        cases.push(values);
    }
    for concurrent in [false, true] {
        for values in &cases {
            let mut graph = DecodeGraph::new_with_timing(concurrent, false, false).unwrap();
            graph.batch_finite = true;
            // Poisoned padding verifies that only the admitted span is read.
            let backing = DecodeGraph::zeros(values.len() + 4).unwrap();
            unsafe {
                let start = backing.buffer.contents().as_ptr().cast::<f32>();
                start.write(f32::NAN);
                start.add(1).write(f32::INFINITY);
                start.add(2).copy_from_nonoverlapping(values.as_ptr(), values.len());
                start.add(values.len() + 2).write(f32::NAN);
                start.add(values.len() + 3).write(f32::NEG_INFINITY);
            }
            let input = backing.slice(2, values.len()).unwrap();
            graph.check_finite(&input).unwrap();
            let index = graph.greedy_argmax(&input).unwrap();
            assert_eq!(index.len, 1);
            assert_eq!(graph.finish(&[index]).unwrap(), [vec![first_index(values) as f32]]);
        }
    }
}

#[stark_test]
fn gpu_greedy_rejects_nonwinning_nonfinite_and_retains_unrelated_obligations() {
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for at in [0, 257, 1024, 1030] {
            let mut graph = DecodeGraph::new().unwrap();
            graph.batch_finite = true;
            graph.finite_selective = true;
            let input = graph.upload_into(&vec![1.0; 1031]).unwrap();
            unsafe {
                input.buffer.contents().as_ptr().cast::<u8>().add(input.offset).cast::<f32>().add(at).write(poison);
            }
            let index = graph.greedy_argmax(&input).unwrap();
            assert!(graph.finish(&[index]).is_err(), "greedy reduction concealed a nonfinite logit");
        }
    }
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut graph = DecodeGraph::new().unwrap();
        let input = graph.upload_into(&vec![1.0; 2051]).unwrap();
        unsafe {
            for at in 1024..2048 {
                input.buffer.contents().as_ptr().cast::<u8>().add(input.offset).cast::<f32>().add(at).write(poison);
            }
        }
        let index = graph.greedy_argmax(&input).unwrap();
        assert!(graph.finish(&[index]).is_err(), "greedy readout concealed an all-poisoned tile");
    }
    let mut graph = DecodeGraph::new().unwrap();
    graph.batch_finite = true;
    let unrelated = graph.upload_into(&[0.0]).unwrap();
    unsafe {
        unrelated.buffer.contents().as_ptr().cast::<u8>().add(unrelated.offset).cast::<f32>().write(f32::NAN);
    }
    graph.check_finite(&unrelated).unwrap();
    let input = graph.upload_into(&[0.0, 2.0, 1.0]).unwrap();
    let index = graph.greedy_argmax(&input).unwrap();
    assert!(graph.finish(&[index]).is_err(), "greedy readout dropped an earlier finite obligation");
    let mut recovered = DecodeGraph::new().unwrap();
    let input = recovered.upload_into(&[0.0, 2.0, 1.0]).unwrap();
    let index = recovered.greedy_argmax(&input).unwrap();
    assert_eq!(recovered.finish(&[index]).unwrap(), [vec![1.0]]);
}

#[stark_test]
fn gpu_greedy_one_value_readback_owns_heap_resources_and_drains_drop() {
    for unretained in [false, true] {
        let heap = ScratchHeap::new(8 * 1024 * 1024).unwrap();
        let mut graph = DecodeGraph::new_with_scratch_heap(&heap, unretained).unwrap();
        let input = graph.upload_into(&[1.0, 4.0, 4.0, -1.0]).unwrap();
        let index = graph.greedy_argmax(&input).unwrap();
        let pending = graph.submit(&[index]).unwrap();
        drop(input);
        drop(heap);
        assert_eq!(pending.outputs.len(), 1);
        assert_eq!(pending.outputs[0].len * 4, 4);
        assert_eq!(pending.wait().unwrap(), [vec![1.0]]);

        let mut graph = DecodeGraph::new().unwrap();
        let input = graph.upload_into(&[1.0, 4.0]).unwrap();
        let index = graph.greedy_argmax(&input).unwrap();
        let command = graph.command.clone();
        drop(graph.submit(&[index]).unwrap());
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    }
    let mut graph = DecodeGraph::new().unwrap();
    let allocation = DecodeGraph::zeros(1).unwrap();
    let invalid = DeviceVector {
        buffer: allocation.buffer.clone(),
        offset: 4,
        len: 1,
    };
    assert!(graph.greedy_argmax(&invalid).is_err());
    graph.projection_level = true;
    assert!(graph.greedy_argmax(&allocation).is_err());
}
