// SPDX-License-Identifier: Apache-2.0
// Stark bindings for the MIT-licensed MLX-Serve HC decode2 body port.
// See metal/official_hc_stock.metal, LICENSE.mlx-serve and UPSTREAM.md.
use anyhow::{Result, ensure};
use half::bf16;

use crate::kernels::quant::check_quant;
use crate::metal::{ComputePass, Grid, MetalContext, MslMathMode, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

const SOURCE: &str = include_str!("metal/official_hc_stock.metal");
const ROWS_SOURCE: &str = concat!(
    include_str!("metal/official_hc_stock.metal"),
    "\n",
    include_str!("metal/official_hc_stock_rows.metal"),
);
const H: usize = 2560;
const G: usize = 4;
const R: usize = 320;
const K: usize = H * G;
pub(crate) const PARTS: usize = G * (R + G);

/// These immutable derived buffers belong to one HC weight descriptor and
/// therefore one loaded model. Original checkpoint codes/tensors are untouched.
pub(crate) struct OfficialHcReadWeights {
    down: QuantWeights,
    up: QuantWeights,
    /// Scaled and transposed [K,G], as stock loadHcWeights prepares.
    inject: Option<Tensor>,
    epsilon: Tensor,
}

fn folded(ctx: &MetalContext, source: &Tensor) -> Result<Tensor> {
    ensure!(source.dtype() == DType::BF16, "HC fold source must be BF16");
    let raw: &[bf16] = bytemuck::cast_slice(source.raw_bytes());
    let values: Vec<bf16> =
        raw.iter().map(|x| bf16::from_f32(x.to_f32() * 0.25)).collect();
    Tensor::from_bytes(ctx, bytemuck::cast_slice(&values), source.shape(), DType::BF16)
}

impl OfficialHcReadWeights {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        ctx: &MetalContext,
        norm: &Tensor,
        down: &QuantWeights,
        up: &QuantWeights,
        inject: Option<&QuantWeights>,
        h: usize,
        groups: usize,
        epsilon: f32,
        official: bool,
    ) -> Result<Option<Self>> {
        if !official
            || !super::official_narrow_sigmoid_supported(
                norm, down, up, inject, h, groups, 1,
            )
            || check_quant(down).is_err()
            || check_quant(up).is_err()
            || !up.codes.binding().1.is_multiple_of(16)
        {
            return Ok(None);
        }
        ensure!(epsilon.is_finite() && epsilon >= 0.0, "invalid HC epsilon");
        let down = QuantWeights {
            codes: down.codes.clone(),
            scales: folded(ctx, &down.scales)?,
            biases: folded(ctx, &down.biases)?,
            group_size: 64,
            bits: 8,
        };
        let up = QuantWeights {
            codes: up.codes.clone(),
            scales: up.scales.clone(),
            biases: up.biases.clone(),
            group_size: 64,
            bits: 8,
        };
        let inject = inject
            .map(|w| -> Result<Tensor> {
                // Loader admission required direct BF16 [G,K]. Transpose and fold
                // once. Neither the original weight nor a global cache is mutated.
                let raw: &[bf16] = bytemuck::cast_slice(w.codes.raw_bytes());
                let mut transposed = Vec::with_capacity(K * G);
                for k in 0..K {
                    for g in 0..G {
                        transposed.push(bf16::from_f32(raw[g * K + k].to_f32() * 0.25));
                    }
                }
                Tensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&transposed),
                    &[K, G],
                    DType::BF16,
                )
            })
            .transpose()?;
        Ok(Some(Self {
            down,
            up,
            inject,
            epsilon: Tensor::from_f32(ctx, &[epsilon], &[1])?,
        }))
    }

    pub(crate) fn selected_for(
        &self,
        official: bool,
        rows: usize,
        read_enabled: bool,
    ) -> bool {
        full_read_selected(official, rows, true, read_enabled)
    }

    pub(crate) fn has_inject(&self) -> bool {
        self.inject.is_some()
    }

    /// One read, exactly ND_SOURCE then vectorized U2V_SOURCE. This is a
    /// source-qualified change of projection/reduction order from skinny Q8.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        hyper: &Tensor,
        norm: &Tensor,
        normalized: &Tensor,
        parts: &Tensor,
        mixed: &Tensor,
        inject_gate: &Tensor,
    ) -> Result<()> {
        ensure!(
            hyper.dtype() == DType::BF16
                && hyper.numel() == K
                && norm.dtype() == DType::BF16
                && norm.numel() == K
                && normalized.dtype() == DType::BF16
                && normalized.numel() == K
                && parts.dtype() == DType::F32
                && parts.numel() == PARTS
                && mixed.dtype() == DType::BF16
                && mixed.numel() == H
                && inject_gate.dtype() == DType::BF16
                && inject_gate.numel() == G,
            "official stock HC read requires one exact-geometry row"
        );
        let (nd_name, up_name, nd_groups) = if self.has_inject() {
            ("official_hc_stock_nd_inject", "official_hc_stock_u2v_inject", 164)
        } else {
            ("official_hc_stock_nd_noinject", "official_hc_stock_u2v_noinject", 160)
        };
        let nd = ctx.pipeline_with_math_mode(
            nd_name,
            SOURCE,
            MslVersion::V3_1,
            MslMathMode::Safe,
        )?;
        pass.dispatch_at(
            &nd,
            &[
                hyper.binding(),
                norm.binding(),
                self.inject.as_ref().unwrap_or(norm).binding(),
                self.epsilon.binding(),
                // WR=0: these operands and xs_out are never accessed.
                norm.binding(),
                norm.binding(),
                self.down.codes.binding(),
                self.down.scales.binding(),
                self.down.biases.binding(),
                normalized.binding(),
                parts.binding(),
                normalized.binding(),
            ],
            &[],
            Grid::Threadgroups { groups: (1, nd_groups, 1), threadgroup: (256, 1, 1) },
        )?;
        pass.level_barrier(&[normalized, parts])?;
        let up = ctx.pipeline_with_math_mode(
            up_name,
            SOURCE,
            MslVersion::V3_1,
            MslMathMode::Safe,
        )?;
        pass.dispatch_at(
            &up,
            &[
                normalized.binding(),
                parts.binding(),
                self.up.codes.binding(),
                self.up.scales.binding(),
                self.up.biases.binding(),
                mixed.binding(),
                inject_gate.binding(),
            ],
            &[],
            Grid::Threadgroups { groups: (80, 1, 1), threadgroup: (512, 1, 1) },
        )
    }

    /// Exact short verifier rows. ND already has an independent row axis;
    /// the U2V specialization shares its loaded weights over those rows.
    /// The ordinary one-row entrypoints retain their original source.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_rows(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        hyper: &Tensor,
        norm: &Tensor,
        normalized: &Tensor,
        parts: &Tensor,
        mixed: &Tensor,
        inject_gate: &Tensor,
    ) -> Result<()> {
        ensure!(hyper.numel().is_multiple_of(K), "stock HC row width");
        let rows = hyper.numel() / K;
        ensure!(
            (1..=4).contains(&rows)
                && hyper.dtype() == DType::BF16
                && norm.dtype() == DType::BF16
                && norm.numel() == K
                && normalized.dtype() == DType::BF16
                && normalized.numel() == rows * K
                && parts.dtype() == DType::F32
                && parts.numel() == rows * PARTS
                && mixed.dtype() == DType::BF16
                && mixed.numel() == rows * H
                && inject_gate.dtype() == DType::BF16
                && inject_gate.numel() == rows * G,
            "official stock HC verifier row geometry"
        );
        if rows == 1 {
            return self.read(
                ctx,
                pass,
                hyper,
                norm,
                normalized,
                parts,
                mixed,
                inject_gate,
            );
        }
        let (nd_name, nd_groups) = if self.has_inject() {
            ("official_hc_stock_nd_inject", 164)
        } else {
            ("official_hc_stock_nd_noinject", 160)
        };
        let nd = ctx.pipeline_with_math_mode(
            nd_name,
            SOURCE,
            MslVersion::V3_1,
            MslMathMode::Safe,
        )?;
        pass.dispatch_at(
            &nd,
            &[
                hyper.binding(),
                norm.binding(),
                self.inject.as_ref().unwrap_or(norm).binding(),
                self.epsilon.binding(),
                norm.binding(),
                norm.binding(),
                self.down.codes.binding(),
                self.down.scales.binding(),
                self.down.biases.binding(),
                normalized.binding(),
                parts.binding(),
                normalized.binding(),
            ],
            &[],
            Grid::Threadgroups {
                groups: (1, nd_groups, rows),
                threadgroup: (256, 1, 1),
            },
        )?;
        pass.level_barrier(&[normalized, parts])?;
        let up_name = match (rows, self.has_inject()) {
            (2, false) => "official_hc_stock_u2v_rows2_noinject",
            (2, true) => "official_hc_stock_u2v_rows2_inject",
            (3, false) => "official_hc_stock_u2v_rows3_noinject",
            (3, true) => "official_hc_stock_u2v_rows3_inject",
            (4, false) => "official_hc_stock_u2v_rows4_noinject",
            (4, true) => "official_hc_stock_u2v_rows4_inject",
            _ => unreachable!(),
        };
        let up = ctx.pipeline_with_math_mode(
            up_name,
            ROWS_SOURCE,
            MslVersion::V3_1,
            MslMathMode::Safe,
        )?;
        pass.dispatch_at(
            &up,
            &[
                normalized.binding(),
                parts.binding(),
                self.up.codes.binding(),
                self.up.scales.binding(),
                self.up.biases.binding(),
                mixed.binding(),
                inject_gate.binding(),
            ],
            &[],
            Grid::Threadgroups { groups: (80, 1, 1), threadgroup: (512, 1, 1) },
        )
    }
}

/// Pure route decision shared by the production descriptor and CPU regression
/// controls. A skipped read cannot produce a finished injection gate.
fn full_read_selected(
    official: bool,
    rows: usize,
    prepared: bool,
    read_enabled: bool,
) -> bool {
    official && rows == 1 && prepared && read_enabled
}

/// The read contract explicitly distinguishes native raw logits from the
/// finished gate returned by stock U2V. Never sigmoid a finished gate twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InjectionInputKind {
    RawLogits,
    PrecomputedGate,
}

impl InjectionInputKind {
    pub(crate) fn from_full_read(selected: bool) -> Self {
        if selected { Self::PrecomputedGate } else { Self::RawLogits }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn inject_after_read(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    hyper: &Tensor,
    branch: &Tensor,
    injection: &Tensor,
    h: usize,
    groups: usize,
    official: bool,
    primitive_sigmoid: bool,
    kind: InjectionInputKind,
) -> Result<()> {
    match kind {
        InjectionInputKind::RawLogits => super::hc_inject_bf16_with_rounding(
            ctx,
            pass,
            hyper,
            branch,
            injection,
            h,
            groups,
            official,
            primitive_sigmoid,
        ),
        InjectionInputKind::PrecomputedGate => {
            ensure!(hyper.numel().is_multiple_of(K), "stock HC gate write row width");
            let rows = hyper.numel() / K;
            ensure!(
                official
                    && !primitive_sigmoid
                    && h == H
                    && groups == G
                    && (1..=4).contains(&rows)
                    && hyper.dtype() == DType::BF16
                    && branch.dtype() == DType::BF16
                    && branch.numel() == rows * H
                    && injection.dtype() == DType::BF16
                    && injection.numel() == rows * G,
                "stock HC gate write requires matching admitted rows"
            );
            let (name, source) = if rows == 1 {
                ("official_hc_write_gate_m1", SOURCE)
            } else {
                ("official_hc_write_gate_exact_rows", ROWS_SOURCE)
            };
            let pipeline = ctx.pipeline(name, source, MslVersion::V3_1)?;
            pass.dispatch_at(
                &pipeline,
                &[hyper.binding(), branch.binding(), injection.binding()],
                &[],
                Grid::Threads { grid: (K, rows, 1), threadgroup: (256, 1, 1) },
            )
        }
    }
}

#[path = "official_hc_stock_pending.rs"]
mod pending;
pub(crate) use pending::{PendingHcWrite, write_gate_to, write_pair_selected};

#[cfg(test)]
#[path = "official_hc_stock_tests.rs"]
mod tests;
