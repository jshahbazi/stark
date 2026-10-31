// SPDX-License-Identifier: Apache-2.0
//! Opt-in synthetic GPU controls for the archived GDN prework prototype.
//! These tests are ignored because another session may own the GPU. They are
//! not production admission, checkpoint parity, or model throughput evidence.
#![cfg(target_os = "macos")]

use anyhow::{Result, ensure};
use stark_lily_core::kernels::gdn::{
    GdnStateLayout, conv1d_step_with_layout, gdn_step_raw_value_key_mlx,
};
use stark_lily_core::metal::{
    ComputePass, Grid, MetalContext, MslMathMode, MslVersion,
};
use stark_lily_core::sha256::{hex, sha256};
use stark_lily_core::tensor::{DType, Tensor};

const C: usize = 10240;
const H: usize = 48;
const D: usize = 128;
const GUARD: usize = 64;
const STEPS: usize = 16;
const PROTOTYPE: &str = include_str!(
    "../../../docs/archive/experiments/qwen38-gdn-prework-2026-10-31/prepared.metal"
);
const PRODUCTION: &str = include_str!("../src/kernels/metal/gdn.metal");

// This test-only wrapper reads the separately materialized production conv.
// Its helper calls bind the production definitions, not prototype definitions.
// It adds no independent stock oracle qualification beyond the named source.
const REFERENCE_PREPARE: &str = r#"
kernel void qwen38_gdn_reference_prepare_test(
    device const bfloat* conv [[buffer(0)]],
    device const bfloat* a [[buffer(1)]],
    device const bfloat* b [[buffer(2)]],
    device const float* A_log [[buffer(3)]],
    device const bfloat* dt [[buffer(4)]],
    device bfloat* prepared [[buffer(5)]],
    device float* gates [[buffer(6)]],
    constant float& k_scale_f32 [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint HK = 16, DIMENSION = 128;
    const uint base = tg * DIMENSION + lane * 4;
    bfloat act[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        act[i] = conv[base + i];
        float v = float(act[i]);
        sumsq += v * v;
    }
    if (tg < 2 * HK) {
        sumsq = simd_sum(sumsq);
        float inv = metal::precise::rsqrt(sumsq / float(DIMENSION) + 1e-6f);
        const bfloat scale = tg < HK ? bfloat(1.0f / float(DIMENSION)) : bfloat(k_scale_f32);
        for (uint i = 0; i < 4; ++i)
            prepared[base + i] = official_gdn_scaled_rms(float(act[i]), inv, scale);
    } else {
        for (uint i = 0; i < 4; ++i) prepared[base + i] = act[i];
        if (lane == 31) {
            const uint head = tg - 2 * HK;
            const bfloat bv = b[head];
            gates[head * 2 + 1] = official_gdn_k1_sigmoid_f32<0>(float(bv));
            const float apd = float(a[head]) + float(dt[head]);
            float sp = gdn_value_key_stock_log1p(metal::precise::exp(apd));
            float ea = metal::precise::exp(A_log[head]);
            gates[head * 2] = metal::precise::exp(-(ea * sp));
        }
    }
}
"#;

#[derive(Clone, Copy)]
enum Variant {
    Staged,
    Direct,
}

impl Variant {
    fn entry(self) -> &'static str {
        match self {
            Self::Staged => "qwen38_gdn_recurrence_staged_research",
            Self::Direct => "qwen38_gdn_recurrence_research",
        }
    }
}

struct Guarded {
    backing: Tensor,
    tensor: Tensor,
}

impl Guarded {
    fn new(ctx: &MetalContext, shape: &[usize], dtype: DType) -> Result<Self> {
        let count: usize = shape.iter().product();
        let bytes = vec![0xa5; (count + 2 * GUARD) * dtype.size()];
        let backing = Tensor::from_bytes(ctx, &bytes, &[count + 2 * GUARD], dtype)?;
        let tensor = backing.view(GUARD, shape)?;
        Ok(Self { backing, tensor })
    }

    fn check(&self, category: &str) -> Result<()> {
        let bytes = self.backing.raw_bytes();
        let edge = GUARD * self.tensor.dtype().size();
        ensure!(
            bytes[..edge]
                .iter()
                .chain(&bytes[bytes.len() - edge..])
                .all(|b| *b == 0xa5),
            "{category}: scratch guard changed"
        );
        ensure!(
            self.tensor.to_f32()?.iter().all(|v| v.is_finite()),
            "{category}: nonfinite output"
        );
        Ok(())
    }

    fn poison(&self) -> Result<()> {
        let word: &[u8] = match self.tensor.dtype() {
            DType::BF16 => &[0xc1, 0x7f],
            DType::F32 => &[0xad, 0xde, 0xc0, 0x7f],
            _ => unreachable!(),
        };
        self.tensor.write_bytes(&word.repeat(self.tensor.numel()))
    }
}

struct Arm {
    window: Guarded,
    state: Guarded,
    conv: Guarded,
    prepared: Guarded,
    gates: Guarded,
    raw: Guarded,
}

impl Arm {
    fn new(ctx: &MetalContext, fixture: usize) -> Result<Self> {
        let arm = Self {
            window: Guarded::new(ctx, &[C, 3], DType::BF16)?,
            state: Guarded::new(ctx, &[H, D, D], DType::F32)?,
            conv: Guarded::new(ctx, &[C], DType::BF16)?,
            prepared: Guarded::new(ctx, &[C], DType::BF16)?,
            gates: Guarded::new(ctx, &[H * 2], DType::F32)?,
            raw: Guarded::new(ctx, &[H, D], DType::BF16)?,
        };
        arm.reset(fixture)?;
        Ok(arm)
    }

    fn reset(&self, fixture: usize) -> Result<()> {
        let history = (0..3 * C)
            .map(|i| (((i * 7 + 3) % 59) as i32 - 29) as f32 / 47.0)
            .collect::<Vec<_>>();
        self.window.tensor.write_bytes(&bf_bytes(&history))?;
        let state = (0..H * D * D)
            .map(|i| {
                if fixture == 0 {
                    0.0
                } else {
                    (((i * 17 + 3) % 257) as i32 - 128) as f32 / 997.0
                        + (i as i32 % 11 - 5) as f32 / 65537.0
                }
            })
            .collect::<Vec<_>>();
        self.state.tensor.write_bytes(bytemuck::cast_slice(&state))?;
        self.poison_outputs()
    }

    fn poison_outputs(&self) -> Result<()> {
        for tensor in [&self.conv, &self.prepared, &self.gates, &self.raw] {
            tensor.poison()?;
        }
        Ok(())
    }

    fn check(&self, reference: bool) -> Result<()> {
        self.window.check("history")?;
        self.state.check("state")?;
        self.prepared.check("prepared QKV")?;
        self.gates.check("gate pairs")?;
        self.raw.check("raw output")?;
        if reference {
            self.conv.check("materialized convolution")?;
        }
        Ok(())
    }
}

struct Inputs {
    weights: Tensor,
    x: Tensor,
    a: Tensor,
    b: Tensor,
    a_log: Tensor,
    dt: Tensor,
}

const EDGES: [u16; 33] = [
    0xc0db, 0x0000, 0x8000, 0x3ebf, 0xbebf, 0x3ec0, 0xbec0, 0x3ec1, 0xbec1, 0x3f9f,
    0xbf9f, 0x3fa0, 0xbfa0, 0x3fff, 0xbfff, 0x4000, 0xc000, 0x4001, 0xc001, 0x401f,
    0xc01f, 0x4020, 0xc020, 0x4021, 0xc021, 0x40ff, 0xc0ff, 0x4100, 0xc100, 0x4180,
    0xc180, 0x42a0, 0xc2a0,
];

impl Inputs {
    fn new(ctx: &MetalContext, fixture: usize, step: usize) -> Result<Self> {
        let weights = (0..4 * C)
            .map(|i| {
                if fixture == 2 {
                    if i / C == 3 { 1.0 } else { 0.0 }
                } else {
                    (((i % C * 3 + i / C * 11) % 31) as i32 - 15) as f32 / 37.0
                }
            })
            .collect::<Vec<_>>();
        let x = (0..C)
            .map(|i| {
                if fixture == 2 {
                    half::bf16::from_bits(EDGES[(i + step) % EDGES.len()]).to_f32()
                } else {
                    (((i * 13 + step * 7) % 113) as i32 - 56) as f32 / 53.0
                }
            })
            .collect::<Vec<_>>();
        let a = (0..H)
            .map(|h| {
                if fixture == 2 {
                    [-80.0, -16.0, -0.00390625, 0.0, 0.00390625, 16.0, 80.0]
                        [(h + step) % 7]
                } else {
                    (h as i32 % 7 - 3) as f32 / 5.0 + step as f32 / 127.0
                }
            })
            .collect::<Vec<_>>();
        let b = (0..H)
            .map(|h| half::bf16::from_bits(EDGES[(h + step) % EDGES.len()]).to_f32())
            .collect::<Vec<_>>();
        let a_log = (0..H)
            .map(|h| half::bf16::from_f32((h as i32 % 9 - 4) as f32 / 13.0).to_f32())
            .collect::<Vec<_>>();
        let dt = (0..H).map(|h| (h as i32 % 5 - 2) as f32 / 17.0).collect::<Vec<_>>();
        Ok(Self {
            weights: Tensor::from_f32_as_bf16(ctx, &weights, &[4, C])?,
            x: Tensor::from_f32_as_bf16(ctx, &x, &[C])?,
            a: Tensor::from_f32_as_bf16(ctx, &a, &[H])?,
            b: Tensor::from_f32_as_bf16(ctx, &b, &[H])?,
            a_log: Tensor::from_f32(ctx, &a_log, &[H])?,
            dt: Tensor::from_f32_as_bf16(ctx, &dt, &[H])?,
        })
    }
}

fn bf_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_ne_bytes())
        .collect()
}

fn exact(a: &Tensor, b: &Tensor, category: &str) -> Result<()> {
    ensure!(a.shape() == b.shape() && a.dtype() == b.dtype(), "{category}: geometry");
    let mismatches = a
        .raw_bytes()
        .chunks_exact(a.dtype().size())
        .zip(b.raw_bytes().chunks_exact(b.dtype().size()))
        .filter(|(a, b)| a != b)
        .count();
    ensure!(mismatches == 0, "{category}: mismatched_elements={mismatches}");
    Ok(())
}

fn reference(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    arm: &Arm,
    input: &Inputs,
) -> Result<()> {
    conv1d_step_with_layout(
        ctx,
        pass,
        &arm.window.tensor,
        &input.x,
        &input.weights,
        &arm.conv.tensor,
        true,
        GdnStateLayout::ValueKey,
    )?;
    pass.level_barrier(&[&arm.window.tensor, &arm.conv.tensor])?;
    let source = format!("{PRODUCTION}\n{REFERENCE_PREPARE}");
    let pipeline = ctx.pipeline_with_math_mode(
        "qwen38_gdn_reference_prepare_test",
        &source,
        MslVersion::V3_1,
        MslMathMode::Safe,
    )?;
    pass.dispatch_at(
        &pipeline,
        &[
            arm.conv.tensor.binding(),
            input.a.binding(),
            input.b.binding(),
            input.a_log.binding(),
            input.dt.binding(),
            arm.prepared.tensor.binding(),
            arm.gates.tensor.binding(),
        ],
        &[&(1.0f32 / 128.0f32.sqrt()).to_ne_bytes()],
        Grid::Threadgroups { groups: (80, 1, 1), threadgroup: (32, 1, 1) },
    )?;
    // Separate the diagnostic stores from the unchanged production recurrence.
    pass.level_barrier(&[&arm.prepared.tensor, &arm.gates.tensor])?;
    gdn_step_raw_value_key_mlx(
        ctx,
        pass,
        &arm.conv.tensor,
        &input.a,
        &input.b,
        &input.a_log,
        &input.dt,
        &arm.state.tensor,
        &arm.raw.tensor,
        1.0f32 / 128.0f32.sqrt(),
        16,
    )
}

fn candidate(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    arm: &Arm,
    input: &Inputs,
    variant: Variant,
) -> Result<()> {
    let prepare = ctx.pipeline_with_math_mode(
        "qwen38_gdn_conv_prepare_research",
        PROTOTYPE,
        MslVersion::V3_1,
        MslMathMode::Safe,
    )?;
    pass.dispatch_at(
        &prepare,
        &[
            arm.window.tensor.binding(),
            input.x.binding(),
            input.weights.binding(),
            arm.prepared.tensor.binding(),
            input.a.binding(),
            input.b.binding(),
            input.a_log.binding(),
            input.dt.binding(),
            arm.gates.tensor.binding(),
        ],
        &[&(1.0f32 / 128.0f32.sqrt()).to_ne_bytes()],
        Grid::Threadgroups { groups: (80, 1, 1), threadgroup: (32, 1, 1) },
    )?;
    pass.level_barrier(&[&arm.window.tensor, &arm.prepared.tensor, &arm.gates.tensor])?;
    let recurrence = ctx.pipeline_with_math_mode(
        variant.entry(),
        PROTOTYPE,
        MslVersion::V3_1,
        MslMathMode::Safe,
    )?;
    pass.dispatch_at(
        &recurrence,
        &[
            arm.prepared.tensor.binding(),
            arm.gates.tensor.binding(),
            arm.state.tensor.binding(),
            arm.raw.tensor.binding(),
        ],
        &[],
        Grid::Threadgroups { groups: (192, 1, 1), threadgroup: (256, 1, 1) },
    )
}

fn correctness(ctx: &MetalContext) -> Result<()> {
    for fixture in 0..3 {
        for variant in [Variant::Staged, Variant::Direct] {
            let baseline = Arm::new(ctx, fixture)?;
            let trial = Arm::new(ctx, fixture)?;
            for step in 0..STEPS {
                baseline.poison_outputs()?;
                trial.poison_outputs()?;
                let input = Inputs::new(ctx, fixture, step)?;
                let pass = ctx.begin_concurrent()?;
                reference(ctx, &pass, &baseline, &input)?;
                pass.level_barrier(&[&baseline.state.tensor, &baseline.raw.tensor])?;
                candidate(ctx, &pass, &trial, &input, variant)?;
                pass.commit()?.wait()?;
                baseline.check(true)?;
                trial.check(false)?;
                exact(&baseline.window.tensor, &trial.window.tensor, "history")?;
                exact(
                    &baseline.prepared.tensor,
                    &trial.prepared.tensor,
                    "prepared QKV",
                )?;
                exact(&baseline.gates.tensor, &trial.gates.tensor, "gate pairs")?;
                exact(
                    &baseline.state.tensor,
                    &trial.state.tensor,
                    "F32 recurrent state",
                )?;
                exact(&baseline.raw.tensor, &trial.raw.tensor, "BF16 raw output")?;
            }
        }
    }
    println!(
        "synthetic_gdn_prework_exact fixtures=3 variants=2 steps_per_chain={STEPS} checks=480"
    );
    Ok(())
}

#[test]
#[ignore = "requires an explicitly available GPU slot; synthetic prototype only"]
fn prepared_gdn_matches_materialized_production_chain_exactly() -> Result<()> {
    correctness(&MetalContext::new()?)
}

#[test]
#[ignore = "requires an explicitly available GPU slot; component timing only"]
fn prepared_gdn_component_abba_after_exactness() -> Result<()> {
    let ctx = MetalContext::new()?;
    correctness(&ctx)?;
    let input = Inputs::new(&ctx, 1, 3)?;
    let arms = (0..36).map(|_| Arm::new(&ctx, 1)).collect::<Result<Vec<_>>>()?;
    // Reference timing excludes the diagnostic preparation dispatch entirely.
    // All three arms run 36 serialized conv+recurrence pairs with independent
    // state/history, fresh identical buffers for each sample, no projections.
    let mut times = [Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()];
    let mut digests = [String::new(), String::new(), String::new()];
    for cycle in 0..4 {
        for arm_index in [0, 1, 2, 2, 1, 0] {
            for arm in &arms {
                arm.reset(1)?;
            }
            let pass = ctx.begin_concurrent()?;
            for arm in &arms {
                if arm_index == 0 {
                    conv1d_step_with_layout(
                        &ctx,
                        &pass,
                        &arm.window.tensor,
                        &input.x,
                        &input.weights,
                        &arm.conv.tensor,
                        true,
                        GdnStateLayout::ValueKey,
                    )?;
                    pass.level_barrier(&[&arm.window.tensor, &arm.conv.tensor])?;
                    gdn_step_raw_value_key_mlx(
                        &ctx,
                        &pass,
                        &arm.conv.tensor,
                        &input.a,
                        &input.b,
                        &input.a_log,
                        &input.dt,
                        &arm.state.tensor,
                        &arm.raw.tensor,
                        1.0f32 / 128.0f32.sqrt(),
                        16,
                    )?;
                } else {
                    candidate(
                        &ctx,
                        &pass,
                        arm,
                        &input,
                        if arm_index == 1 { Variant::Staged } else { Variant::Direct },
                    )?;
                }
                pass.level_barrier(&[
                    &arm.window.tensor,
                    &arm.state.tensor,
                    &arm.raw.tensor,
                ])?;
            }
            let completed = pass.commit()?.wait_retain()?;
            let timing = completed.timing()?;
            let seconds = timing.gpu_end_secs - timing.gpu_start_secs;
            ensure!(seconds.is_finite() && seconds > 0.0, "invalid component timing");
            if cycle > 0 {
                times[arm_index].push(seconds / 36.0);
            }
            let mut identity = Vec::new();
            for arm in &arms {
                arm.window.check("timed history")?;
                arm.state.check("timed state")?;
                arm.raw.check("timed raw")?;
                identity.extend_from_slice(arm.state.tensor.raw_bytes());
                identity.extend_from_slice(arm.raw.tensor.raw_bytes());
                identity.extend_from_slice(arm.window.tensor.raw_bytes());
            }
            let digest = hex(&sha256(&identity));
            if digests[arm_index].is_empty() {
                digests[arm_index] = digest;
            } else {
                ensure!(
                    digests[arm_index] == digest,
                    "timed arm repeat digest changed"
                );
            }
        }
    }
    ensure!(
        digests[0] == digests[1] && digests[0] == digests[2],
        "timed cross-arm digest differs"
    );
    for (index, name) in
        ["materialized", "prepared_staged", "prepared_direct"].iter().enumerate()
    {
        let mean = times[index].iter().sum::<f64>() / times[index].len() as f64;
        println!(
            "gdn_component arm={name} samples={} mean_gpu_us={:.6} digest={}",
            times[index].len(),
            mean * 1e6,
            digests[index]
        );
    }
    Ok(())
}
