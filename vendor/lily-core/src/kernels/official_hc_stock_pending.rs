// SPDX-License-Identifier: Apache-2.0
// Stark pending-write bindings for the pinned MIT MLX-Serve ND_SOURCE.
// Original WR0 bindings and their fallback remain unchanged.
use super::*;

const PENDING_SOURCE: &str = include_str!("metal/official_hc_stock_pending.metal");

/// Incoming gate is the finished BF16 2*sigmoid output, never raw logits.
#[derive(Clone, Copy)]
pub(crate) struct PendingHcWrite<'a> {
    pub(crate) branch: &'a Tensor,
    pub(crate) gate: &'a Tensor,
    pub(crate) kind: InjectionInputKind,
}

/// Both selected reads already include policy, geometry, row and HC-ablation
/// admission. No write may be deferred when injection or either read is skipped.
pub(crate) const fn write_pair_selected(
    first_read: bool,
    next_read: bool,
    first_inject: bool,
    next_inject: bool,
    injection_enabled: bool,
    scratch: bool,
) -> bool {
    first_read
        && next_read
        && first_inject
        && next_inject
        && injection_enabled
        && scratch
}

fn span(t: &Tensor) -> Result<(usize, usize)> {
    let begin = t.identity();
    let end = begin
        .checked_add(t.byte_len())
        .ok_or_else(|| anyhow::anyhow!("HC span overflow"))?;
    Ok((begin, end))
}

fn overlaps(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

impl OfficialHcReadWeights {
    /// Stock WR1 ND followed by the unchanged U2V read. All ND groups read
    /// the old stream: written MUST be disjoint, not an in-place alias.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_after_write(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        hyper: &Tensor,
        norm: &Tensor,
        normalized: &Tensor,
        parts: &Tensor,
        mixed: &Tensor,
        inject_gate: &Tensor,
        written: &Tensor,
        pending: PendingHcWrite<'_>,
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
                && inject_gate.numel() == G
                && written.dtype() == DType::BF16
                && written.numel() == K
                && pending.branch.dtype() == DType::BF16
                && pending.branch.numel() == H
                && pending.gate.dtype() == DType::BF16
                && pending.gate.numel() == G
                && pending.kind == InjectionInputKind::PrecomputedGate,
            "stock pending HC read requires one admitted row and a finished gate"
        );
        let written_span = span(written)?;
        for tensor in [
            hyper,
            norm,
            normalized,
            parts,
            mixed,
            inject_gate,
            pending.branch,
            pending.gate,
            &self.down.codes,
            &self.down.scales,
            &self.down.biases,
            &self.up.codes,
            &self.up.scales,
            &self.up.biases,
            &self.epsilon,
        ] {
            ensure!(
                !overlaps(written_span, span(tensor)?),
                "pending HC stream output overlaps another operand"
            );
        }
        if let Some(inject) = &self.inject {
            ensure!(
                !overlaps(written_span, span(inject)?),
                "pending HC stream output overlaps inject weights"
            );
        }
        for tensor in [pending.branch, pending.gate] {
            let input = span(tensor)?;
            ensure!(
                !overlaps(input, span(normalized)?) && !overlaps(input, span(parts)?),
                "pending HC input overlaps an ND output"
            );
        }
        let (nd_name, up_name, groups) = if self.has_inject() {
            ("official_hc_stock_nd_pending_inject", "official_hc_stock_u2v_inject", 164)
        } else {
            (
                "official_hc_stock_nd_pending_noinject",
                "official_hc_stock_u2v_noinject",
                160,
            )
        };
        let nd = ctx.pipeline_with_math_mode(
            nd_name,
            PENDING_SOURCE,
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
                pending.branch.binding(),
                pending.gate.binding(),
                self.down.codes.binding(),
                self.down.scales.binding(),
                self.down.biases.binding(),
                normalized.binding(),
                parts.binding(),
                written.binding(),
            ],
            &[],
            Grid::Threadgroups { groups: (1, groups, 1), threadgroup: (256, 1, 1) },
        )?;
        // U2V may now overwrite the same injection buffer whose old gate ND
        // consumed. The caller cannot overwrite branch until this read returns.
        pass.level_barrier(&[normalized, parts, written])?;
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
}

/// The pair finishes in canonical hyper storage. This replaces the existing
/// MoE gate-write dispatch; it does not add a copy or defer state across layers.
pub(crate) fn write_gate_to(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    residual: &Tensor,
    output: &Tensor,
    official: bool,
    pending: PendingHcWrite<'_>,
) -> Result<()> {
    ensure!(
        official
            && residual.dtype() == DType::BF16
            && residual.numel() == K
            && output.dtype() == DType::BF16
            && output.numel() == K
            && pending.branch.dtype() == DType::BF16
            && pending.branch.numel() == H
            && pending.gate.dtype() == DType::BF16
            && pending.gate.numel() == G
            && pending.kind == InjectionInputKind::PrecomputedGate,
        "stock HC output write requires one admitted row and a finished gate"
    );
    let (input, out) = (span(residual)?, span(output)?);
    ensure!(
        input == out || !overlaps(input, out),
        "HC output write partially overlaps residual"
    );
    ensure!(
        !overlaps(out, span(pending.branch)?) && !overlaps(out, span(pending.gate)?),
        "HC output write overlaps branch or gate"
    );
    let pipeline =
        ctx.pipeline("official_hc_write_gate_to_m1", PENDING_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            residual.binding(),
            output.binding(),
            pending.branch.binding(),
            pending.gate.binding(),
        ],
        &[],
        Grid::Threads { grid: (K, 1, 1), threadgroup: (256, 1, 1) },
    )
}
