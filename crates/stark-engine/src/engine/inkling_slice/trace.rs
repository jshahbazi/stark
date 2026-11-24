// SPDX-License-Identifier: Apache-2.0
//! Bounded diagnostic copies into shared GPU buffers. The fixture supplies the
//! reference values. The diagnostic publishes aggregate comparisons only.

use super::*;

pub(super) const CONTROLLER_PAIR: [&str; 4] =
    ["controller_pair_0", "controller_pair_1", "controller_pair_2", "controller_pair_3"];
pub(super) const CONTROLLER_RANK: [&str; 4] =
    ["controller_rank_0", "controller_rank_1", "controller_rank_2", "controller_rank_3"];
pub(super) const CONTROLLER_OUT: [&str; 4] =
    ["controller_out_0", "controller_out_1", "controller_out_2", "controller_out_3"];
pub(super) const ATTN_NORM: [&str; 3] = ["attn_norm_0", "attn_norm_1", "attn_norm_2"];
pub(super) const RAW_PROJECTIONS: [&str; 6] = ["q_raw", "k_raw", "v_raw", "r_raw", "gate_raw", "sink_raw"];

pub(super) struct Point {
    pub(super) name: &'static str,
    pub(super) values: usize,
    pub(super) dtype: &'static str,
    pub(super) buffer: Buffer,
}

pub(super) struct Trace {
    pub(super) points: Vec<Point>,
}

impl Trace {
    pub(super) fn allocate(
        device: &Device,
        fixture: &Tensors,
        physical: usize,
    ) -> Result<Option<Self>> {
        let mut specs = vec![
            (
                "attn_mixed",
                "F32",
                if physical == 10 {
                    WIDTH
                } else {
                    3 * WIDTH
                },
            ),
            ("mlp_mixed", "F32", WIDTH),
            ("attn_norm_0", "BF16", WIDTH),
            ("attn_norm_1", "BF16", WIDTH),
            ("attn_norm_2", "BF16", WIDTH),
            ("q_raw", "BF16", 3072),
            ("k_raw", "BF16", KV),
            ("v_raw", "BF16", KV),
            ("r_raw", "BF16", HEADS * 16),
            ("gate_raw", "BF16", 3072),
            ("sink_raw", "BF16", HEADS),
            ("attended", "BF16", 3072),
            ("attn_raw", "BF16", WIDTH),
            ("router_logits", "F32", EXPERTS),
        ];
        for i in 0..4 {
            specs.extend([
                (CONTROLLER_PAIR[i], "BF16", 2 * WIDTH),
                (CONTROLLER_RANK[i], "BF16", 64),
                (CONTROLLER_OUT[i], "F32", WIDTH),
            ]);
        }
        let mut points = Vec::new();
        for (name, dtype, values) in specs {
            if fixture.contains(&format!("{physical}.trace.{name}")) {
                points.push(Point {
                    name,
                    values,
                    dtype,
                    buffer: zeroed(
                        device,
                        values
                            * if dtype == "BF16" {
                                2
                            } else {
                                4
                            },
                    )?,
                });
            }
        }
        Ok((!points.is_empty()).then_some(Self {
            points,
        }))
    }

    pub(super) fn encode(
        &self,
        gpu: &mut Stream<'_>,
        name: &str,
        source: &Buffer,
    ) {
        if let Some(point) = self.points.iter().find(|point| point.name == name) {
            let words = point.buffer.length() / 4;
            gpu.one("ink_trace_copy", &[source, &point.buffer], words as u32, (words.div_ceil(128), 1), 128);
        }
    }
}
