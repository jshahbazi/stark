// SPDX-License-Identifier: Apache-2.0
//! Serial finite/prepass ordering keeps concurrent dependencies explicit.
use stark_engine_macros::stark_test;

use super::*;

fn graph(
    concurrent: bool,
    enabled: bool,
) -> DecodeGraph {
    let mut graph = DecodeGraph::new_with_timing(concurrent, false, false).unwrap();
    graph.batch_finite = true;
    graph.finite_selective = true;
    graph.serial_finite_order = enabled;
    graph.serial_projection_levels = false;
    graph
}

#[stark_test]
fn serial_finite_order_uses_actual_dispatch_type_and_preserves_bits() {
    for concurrent in [false, true] {
        for enabled in [false, true] {
            let mut graph = graph(concurrent, enabled);
            assert_eq!(graph.serial_finite_dispatch_order(), enabled && !concurrent);
            let input = graph.upload_into(&[1.0, -2.0, 3.0]).unwrap();
            let first = graph.add(&input, &input).unwrap();
            graph.check_finite(&first).unwrap();
            graph.begin_projection_level().unwrap();
            assert_eq!(graph.encoding_dispatch, MTLDispatchType::CONCURRENT);
            assert!(!graph.serial_finite_dispatch_order());
            let second = graph.add(&first, &first).unwrap();
            graph.check_finite(&second).unwrap();
            graph.end_projection_level().unwrap();
            graph.resource_boundary().unwrap();
            assert_eq!(graph.serial_finite_dispatch_order(), enabled && !concurrent);
            graph.copy_into(&input, &first).unwrap();
            let output = graph.add(&first, &second).unwrap();
            graph.check_finite(&output).unwrap();
            assert_eq!(graph.finish(&[output]).unwrap(), [vec![5.0, -10.0, 15.0]]);
        }
    }
}

#[stark_test]
fn serial_finite_order_preserves_masked_poison_before_overwrites() {
    for concurrent in [false, true] {
        for enabled in [false, true] {
            for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut graph = graph(concurrent, enabled);
                let target = DecodeGraph::zeros(1031).unwrap();
                // SAFETY: fresh owned test storage has no GPU users.
                unsafe {
                    target.buffer.contents().as_ptr().cast::<f32>().add(1030).write(poison);
                }
                graph.check_finite(&target).unwrap();
                graph.resource_boundary().unwrap();
                graph.copy_into(&DecodeGraph::upload(&vec![0.0; 1031]).unwrap(), &target).unwrap();
                assert!(graph.finish(&[target]).is_err());
            }
        }
    }
}

#[stark_test]
fn serial_finite_order_preserves_gpu_input_bound_dependency_and_failure() {
    for concurrent in [false, true] {
        for enabled in [false, true] {
            for projection in [false, true] {
                for poisoned in [false, true] {
                    let mut graph = graph(concurrent, enabled);
                    let mut values = vec![0.25; 1024];
                    values[1023] = if poisoned {
                        f32::NAN
                    } else {
                        1e30
                    };
                    let input = DecodeGraph::zeros(values.len()).unwrap();
                    // SAFETY: fresh owned test storage has no GPU users.
                    unsafe {
                        input
                            .buffer
                            .contents()
                            .as_ptr()
                            .cast::<f32>()
                            .copy_from_nonoverlapping(values.as_ptr(), values.len());
                    }
                    if projection {
                        graph.begin_projection_level().unwrap();
                    }
                    let bounds = graph.iq_input_bounds(&input, 512, 2).unwrap();
                    let copied = graph.scratch(2).unwrap();
                    graph.copy_into(&bounds, &copied).unwrap();
                    if projection {
                        graph.end_projection_level().unwrap();
                    }
                    graph.resource_boundary().unwrap();
                    let actual = graph.finish(&[copied]);
                    if poisoned {
                        assert!(actual.is_err());
                    } else {
                        assert_eq!(actual.unwrap()[0].iter().map(|x| x.to_bits()).collect::<Vec<_>>(), [0, 1]);
                    }
                }
            }
        }
    }
}
