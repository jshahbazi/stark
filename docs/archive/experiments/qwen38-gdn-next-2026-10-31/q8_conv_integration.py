"""Patch only a private archived source copy with fixed Q8/conv research code."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
METAL_REL = "vendor/lily-core/src/kernels/metal/qwen38_q8_conv_research.metal"
PROJECTION_REL = "vendor/lily-core/src/kernels/official_q8_projection.rs"
MODEL_REL = "vendor/lily-core/src/qwen4exp/model.rs"


def replace_once(source: str, old: str, new: str, category: str) -> str:
    if source.count(old) != 1:
        raise ValueError(category)
    return source.replace(old, new, 1)


WRAPPER = '''

// Research-only fixed official Q8 input projection/convolution path.
// Default shader build does not materialize the unused raw QKV prefix.
const QWEN38_Q8_CONV_RESEARCH: &str = include_str!("metal/qwen38_q8_conv_research.metal");

pub(crate) fn gemv_gdn_conv_research(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    x: &Tensor,
    joined: &Tensor,
    window: &Tensor,
    conv_w: &Tensor,
    convolved: &Tensor,
) -> Result<()> {
    const K: usize = 2560;
    const N: usize = 16480;
    const C: usize = 10240;
    ensure!(w.bits == 8 && w.group_size == 64 && w.codes.shape() == [N, K / 4], "research Q8 conv matrix");
    ensure!(w.scales.dtype() == DType::BF16 && w.biases.dtype() == DType::BF16, "research Q8 conv affine types");
    ensure!(x.numel() == K && x.dtype() == DType::BF16, "research Q8 conv input");
    ensure!(joined.numel() == N && joined.dtype() == DType::BF16, "research Q8 conv joined output");
    ensure!(window.shape() == [C, 3] && window.dtype() == DType::BF16, "research Q8 conv history");
    ensure!(conv_w.shape() == [4, C] && conv_w.dtype() == DType::BF16, "research Q8 conv tap weights");
    ensure!(convolved.numel() == C && convolved.dtype() == DType::BF16, "research Q8 conv output");
    quant::check_quant(w)?;
    ensure!(stock_signed_offsets_fit(N, K), "research Q8 conv signed addressing");
    let pipeline = ctx.pipeline_with_math_mode(
        "qwen38_q8_projection_conv_research",
        QWEN38_Q8_CONV_RESEARCH,
        MslVersion::V4_0,
        MslMathMode::Safe,
    )?;
    ensure!(pipeline.thread_execution_width() == 32, "research Q8 conv SIMD width");
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(), w.scales.binding(), w.biases.binding(), x.binding(),
            joined.binding(), window.binding(), conv_w.binding(), convolved.binding(),
        ],
        &[&u32_bytes(K), &u32_bytes(N)],
        Grid::Threadgroups { groups: (1, N / 8, 1), threadgroup: (32, 2, 1) },
    )
}
'''


def patch_source(source_root: Path) -> None:
    """Called by the parent experiment's source-archive preparer, never a build."""
    source_root = source_root.resolve()
    if source_root == ROOT or not source_root.is_relative_to((ROOT / "runtime").resolve()):
        raise ValueError("research_patch_outside_private_archive")
    if (source_root / ".git").exists():
        raise ValueError("research_patch_git_checkout_refused")
    projection = source_root / PROJECTION_REL
    model = source_root / MODEL_REL
    projection_source = projection.read_text()
    if "QWEN38_Q8_CONV_RESEARCH" in projection_source:
        raise ValueError("research_patch_already_applied")
    model_source = model.read_text()
    anchor = """        // The stacked qkv/z/a/b projection is one dependency group.
"""
    addition = """        let research_q8_conv = self.config.official_mlx_rounding()
            && gdn_layout == GdnStateLayout::ValueKey
            && !crate::ablate::on(crate::ablate::Group::GdnProj)
            && !crate::ablate::on(crate::ablate::Group::GdnStep)
            && w.in_proj.bits == 8
            && w.in_proj.group_size == 64
            && w.in_proj.out_features() == 16480
            && w.in_proj.in_features() == 2560
            && self.config.linear_conv_kernel_dim == 4;
""" + anchor
    model_source = replace_once(model_source, anchor, addition, "research_q8_conv_model_anchor")
    model_source = replace_once(
        model_source,
        "        pass.profile_stage(if fused_conv {\n",
        "        pass.profile_stage(if fused_conv || research_q8_conv {\n",
        "research_q8_conv_profile_anchor",
    )
    model_source = replace_once(
        model_source,
        """        if !crate::ablate::on(crate::ablate::Group::GdnProj) {
            if fused_conv {
""",
        """        if !crate::ablate::on(crate::ablate::Group::GdnProj) {
            if research_q8_conv {
                crate::kernels::official_q8_projection::gemv_gdn_conv_research(
                    ctx, pass, &w.in_proj, &s.hc.mixed, &s.gdn_in,
                    conv_window, &w.conv_w, &s.qkv_conv,
                )?;
                pass.level_barrier(&[&s.gdn_in, &s.qkv_conv, conv_window])?;
            } else if fused_conv {
""",
        "research_q8_conv_dispatch_anchor",
    )
    model_source = replace_once(
        model_source,
        "        if !fused_conv {\n",
        "        if !fused_conv && !research_q8_conv {\n",
        "research_q8_conv_skip_anchor",
    )
    shader = Path(__file__).with_name("q8_conv.metal").read_bytes()
    if b"#define RESEARCH_CAPTURE_RAW_QKV 0" not in shader:
        raise ValueError("research_q8_conv_raw_capture_policy")
    target = source_root / METAL_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(shader)
    projection.write_text(projection_source + WRAPPER)
    model.write_text(model_source)
