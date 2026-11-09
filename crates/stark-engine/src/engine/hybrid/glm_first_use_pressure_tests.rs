// SPDX-License-Identifier: Apache-2.0
//! One ignored, nonmodel first-use discriminator. No mapped files or timing claim.

use std::{
    io::{Read, Write},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use stark_engine_macros::stark_test;

use super::*;
use crate::engine::hybrid::glm_metal::{Control, ControlGeometry};

const PROTOCOL: &str = "glm-first-use-control-q5-v15";
const HIDDEN: usize = 256;
const STREAMS: usize = 4;
const ROWS: usize = 13;

fn mark(
    started: Instant,
    phase: &'static str,
    initialized: bool,
) {
    // Before initialization, querying allocated_bytes() would itself initialize
    // the weight library and erase the phase boundary we intend to observe.
    let allocated = initialized.then(|| allocated_bytes().unwrap());
    println!(
        "STARK_GLM_FIRST_USE15 {}",
        serde_json::json!({
            "protocol": PROTOCOL,
            "phase": phase,
            "pid": std::process::id(),
            "elapsed_ns": started.elapsed().as_nanos() as u64,
            "wall_time_ns": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            "metal_allocated_bytes": allocated,
        })
    );
    std::io::stdout().flush().unwrap();
}

fn check_projection(
    codes: &[u8],
    result: &[Vec<f32>],
) {
    assert_eq!(result.len(), 2, "first-use output count changed");
    assert_eq!(result[0].len(), HIDDEN, "first-use normalized shape changed");
    assert_eq!(result[1].len(), ROWS, "first-use Q5 shape changed");
    assert!(result.iter().flatten().all(|v| v.is_finite()), "first-use output is nonfinite");
    assert!(result[0].iter().any(|v| v.abs() > 1e-6), "first-use normalized input is empty");
    // Existing independent CPU byte decoder plus F64 products, never the GPU
    // projection implementation. The actual Control output is its input.
    let oracle = tests::packed_f64_oracle(13, codes, ROWS, HIDDEN, &result[0], 0);
    assert!(oracle.iter().any(|(value, _)| value.abs() > 1e-6), "first-use projection is empty");
    for (&actual, &(expected, absolute)) in result[1].iter().zip(&oracle) {
        assert!((f64::from(actual) - expected).abs() <= 2e-5 * absolute.max(1.0), "first-use Q5 F64 projection error");
    }
}

#[stark_test]
#[ignore]
fn glm_first_use_control_q5_pressure_control() {
    // A fresh test process and exact --ignored --exact filter are part of the
    // registration. Reject inherited shader selectors before any Metal access.
    assert_eq!(std::env::var("STARK_TEST_GLM_FIRST_USE15").as_deref(), Ok("1"));
    assert!(
        std::env::vars_os().all(|(key, _)| {
            let key = key.to_string_lossy();
            !key.starts_with("STARK_GLM53_") && !key.starts_with("STARK_HYBRID_") && !key.starts_with("STARK_QWEN38_")
        }),
        "first-use shader selectors must be absent"
    );
    let format = MatrixFormat::Packed(13);
    // These are the same family predicates used by pipeline_for_geometry.
    // The actual first model projection has rows8192/columns4096; this tiny
    // shape intentionally keeps the same Safe block kernel and64-thread grid.
    assert!(block_schedule(format) && !factored_schedule(format));
    assert!(!q5_grouped32_shape(format, HIDDEN) && !q5_grouped32_shape(format, 4096));
    assert_eq!(matrix_threads(format, ROWS, HIDDEN), 64);
    assert_eq!(matrix_threads(format, 8192, 4096), 64);
    assert_eq!(matrix_groups(format, ROWS, HIDDEN), ROWS.div_ceil(2));
    assert_eq!(matrix_groups(format, 8192, 4096), 4096);
    let started = Instant::now();
    mark(started, "waiting_for_owner_start", false);
    // The supervisor must capture a live process/service sample before the
    // owner releases this one-byte gate. No Metal API has run at this point.
    let mut gate = [0u8; 1];
    std::io::stdin().read_exact(&mut gate).unwrap();
    assert!(gate == *b"G", "first-use owner start gate is invalid");
    mark(started, "before_weight_library", false);
    initialize().unwrap();
    mark(started, "weight_library_packed0_ready", true);

    objc2::rc::autoreleasepool(|_| {
        let columns = HIDDEN * STREAMS;
        let control_rows = (2 + STREAMS) * STREAMS;
        let control_weights: Vec<_> =
            (0..control_rows * columns).map(|i| (((i * 13 + 17) % 127) as f32 - 63.0) * 0.0001).collect();
        let control = Control::new(
            ControlGeometry {
                hidden: HIDDEN,
                streams: STREAMS,
                iterations: 20,
                eps: 1e-6,
                rms_eps: 1e-5,
            },
            &control_weights,
            &vec![0.0; control_rows],
            &[0.01, 0.01, 0.01],
            &vec![1.0; HIDDEN],
        )
        .unwrap();
        let residual_values: Vec<_> = (0..columns).map(|i| ((i % 127) as f32 - 63.0) * 0.001).collect();
        let residual = DecodeGraph::upload(&residual_values).unwrap();
        let codes = tests::finite_packed_codes(13, ROWS, HIDDEN, 41);
        assert_eq!(codes.len(), 2288, "first-use packed allocation changed");
        // Owned shared buffer, with initialized exact bytes. No NoCopy view,
        // full-file VM mapping or model payload participates in this control.
        let matrix = MappedMatrix {
            codes: MappedSlice {
                buffer: buffer(executor().unwrap(), &codes).unwrap(),
                offset: 0,
                length: codes.len(),
                registered_length: codes.len(),
            },
            planes: None,
        };
        let operation = MappedOperation {
            matrices: vec![matrix],
            inputs: &[],
            rows: ROWS,
            columns: HIDDEN,
            rows_per_batch: 0,
            format,
        };
        mark(started, "control_and_q5_owned_buffers_ready", true);
        let mut graph = DecodeGraph::new().unwrap();
        mark(started, "before_control_pre_encoding", true);
        let normalized = control.pre(&mut graph, &residual).unwrap();
        mark(started, "control_pre_encoded_glm_libraries_psos_ready", true);
        // Do not submit Control separately. Match actual Control→Q5 ordering
        // through the same ordinary retained, serial, non-heap graph transport.
        let output = graph.mapped(&operation, &normalized).unwrap();
        mark(started, "q5_block_pipeline_and_dispatch_encoded", true);
        let result = graph.finish(&[normalized, output]).unwrap();
        mark(started, "first_graph_completed", true);
        check_projection(&codes, &result);
        drop(result);
        for _ in 0..2 {
            let mut graph = DecodeGraph::new().unwrap();
            let normalized = control.pre(&mut graph, &residual).unwrap();
            let output = graph.mapped(&operation, &normalized).unwrap();
            let result = graph.finish(&[normalized, output]).unwrap();
            check_projection(&codes, &result);
        }
        mark(started, "two_warm_graphs_completed", true);
    });
    // Executor/PSO singletons intentionally remain process-owned. Only test
    // buffers and command/autorelease owners have drained at this boundary.
    mark(started, "test_buffer_owners_drained", true);
    mark(started, "control_passed", true);
}
