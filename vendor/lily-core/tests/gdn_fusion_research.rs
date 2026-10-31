// SPDX-License-Identifier: Apache-2.0
//! Ignored synthetic correctness and component controls for the archived
//! ValueKey recurrence/output-norm fusion research shader.
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
const FUSED: &str = include_str!(
    "../../../docs/archive/experiments/qwen38-gdn-fusion-2026-10-31/fused.metal"
);
const PRODUCTION: &str = include_str!("../src/kernels/metal/gdn.metal");

#[derive(Clone, Copy, Debug)]
enum Variant {
    Fused256,
    Fused512,
}

impl Variant {
    fn entry(self) -> &'static str {
        match self {
            Self::Fused256 => "qwen38_gdn_fused256_research",
            Self::Fused512 => "qwen38_gdn_fused512_research",
        }
    }

    fn threads(self) -> usize {
        match self {
            Self::Fused256 => 256,
            Self::Fused512 => 512,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Fused256 => "fused256",
            Self::Fused512 => "fused512",
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
        let backing = Tensor::from_bytes(
            ctx,
            &vec![0xa5; (count + 2 * GUARD) * dtype.size()],
            &[count + 2 * GUARD],
            dtype,
        )?;
        let tensor = backing.view(GUARD, shape)?;
        Ok(Self { backing, tensor })
    }

    fn poison(&self) -> Result<()> {
        let word: &[u8] = match self.tensor.dtype() {
            DType::BF16 => &[0xc1, 0x7f],
            DType::F32 => &[0xad, 0xde, 0xc0, 0x7f],
            _ => unreachable!(),
        };
        self.tensor.write_bytes(&word.repeat(self.tensor.numel()))
    }

    fn check_guards(&self, label: &str) -> Result<()> {
        let bytes = self.backing.raw_bytes();
        let edge = GUARD * self.tensor.dtype().size();
        ensure!(
            bytes[..edge]
                .iter()
                .chain(&bytes[bytes.len() - edge..])
                .all(|b| *b == 0xa5),
            "{label}: guard changed"
        );
        Ok(())
    }

    fn check(&self, label: &str) -> Result<()> {
        self.check_guards(label)?;
        ensure!(
            self.tensor.to_f32()?.iter().all(|v| v.is_finite()),
            "{label}: nonfinite result"
        );
        Ok(())
    }
}

struct Arm {
    window: Guarded,
    state: Guarded,
    conv: Guarded,
    raw: Guarded,
    z: Guarded,
    norm_w: Guarded,
    gated: Guarded,
}

impl Arm {
    fn new(ctx: &MetalContext, fixture: usize) -> Result<Self> {
        let arm = Self {
            window: Guarded::new(ctx, &[C, 3], DType::BF16)?,
            state: Guarded::new(ctx, &[H, D, D], DType::F32)?,
            conv: Guarded::new(ctx, &[C], DType::BF16)?,
            raw: Guarded::new(ctx, &[H, D], DType::BF16)?,
            z: Guarded::new(ctx, &[H, D], DType::BF16)?,
            norm_w: Guarded::new(ctx, &[D], DType::F32)?,
            gated: Guarded::new(ctx, &[H, D], DType::BF16)?,
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
        let z = (0..H * D)
            .map(|i| match fixture {
                2 => {
                    const EDGES: [u16; 12] = [
                        0x0000, 0x8000, 0x3f80, 0xbf80, 0x4120, 0xc120, 0x3eab, 0xbeab,
                        0x42c8, 0xc2c8, 0x7f7f, 0xff7f,
                    ];
                    half::bf16::from_bits(EDGES[(i * 7 + i / D) % EDGES.len()]).to_f32()
                }
                _ => (((i * 13 + fixture * 19) % 113) as i32 - 56) as f32 / 53.0,
            })
            .collect::<Vec<_>>();
        self.z.tensor.write_bytes(&bf_bytes(&z))?;
        let weights = (0..D)
            .map(|i| match fixture {
                2 => [0.0, 1.0, -1.0, 0.333251953125][i % 4],
                _ => 0.75 + (i as i32 % 17 - 8) as f32 / 31.0,
            })
            .collect::<Vec<_>>();
        self.norm_w.tensor.write_bytes(bytemuck::cast_slice(&weights))?;
        self.poison_outputs()
    }

    fn poison_outputs(&self) -> Result<()> {
        for item in [&self.conv, &self.raw, &self.gated] {
            item.poison()?;
        }
        Ok(())
    }

    fn check(&self, check_raw: bool) -> Result<()> {
        self.window.check("history")?;
        self.conv.check("convolved QKV")?;
        self.state.check("state")?;
        self.gated.check("gated output")?;
        self.z.check("z input")?;
        self.norm_w.check("norm weights")?;
        if check_raw {
            self.raw.check("raw output")?;
        } else {
            self.raw.check_guards("unused raw")?;
            ensure!(
                self.raw.tensor.raw_bytes() == [0xc1u8, 0x7f].repeat(H * D),
                "unused raw changed"
            );
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
                    const EDGES: [u16; 12] = [
                        0x0000, 0x8000, 0x3f80, 0xbf80, 0x4120, 0xc120, 0x3eab, 0xbeab,
                        0x42c8, 0xc2c8, 0x3f01, 0xbf01,
                    ];
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
            .map(|h| {
                half::bf16::from_bits(
                    [0x0000, 0x8000, 0x3f80, 0xbf80, 0x4120, 0xc120, 0x3eab]
                        [(h + step) % 7],
                )
                .to_f32()
            })
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

fn exact(a: &Tensor, b: &Tensor, label: &str) -> Result<()> {
    ensure!(
        a.shape() == b.shape() && a.dtype() == b.dtype(),
        "{label}: geometry differs"
    );
    let mismatches = a
        .raw_bytes()
        .chunks_exact(a.dtype().size())
        .zip(b.raw_bytes().chunks_exact(b.dtype().size()))
        .filter(|(x, y)| x != y)
        .count();
    ensure!(mismatches == 0, "{label}: mismatched_elements={mismatches}");
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
    )?;
    pass.level_barrier(&[&arm.state.tensor, &arm.raw.tensor])?;
    let pipeline = ctx.pipeline_with_math_mode(
        "gdn_normgate_value_key_mlx",
        PRODUCTION,
        MslVersion::V3_1,
        MslMathMode::Safe,
    )?;
    pass.dispatch_at(
        &pipeline,
        &[
            arm.raw.tensor.binding(),
            arm.z.tensor.binding(),
            arm.norm_w.tensor.binding(),
            arm.gated.tensor.binding(),
        ],
        &[&1e-6f32.to_ne_bytes()],
        Grid::Threadgroups { groups: (H, 1, 1), threadgroup: (32, 1, 1) },
    )
}

fn fused(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    arm: &Arm,
    input: &Inputs,
    variant: Variant,
    capture_raw: bool,
) -> Result<()> {
    // Convolution stays on the admitted production path. The experimental
    // entry fuses the ValueKey recurrence and output norm/gate only.
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
    let source = if capture_raw {
        format!("#define RESEARCH_CAPTURE_RAW 1\n{FUSED}")
    } else {
        FUSED.to_owned()
    };
    let pipeline = ctx.pipeline_with_math_mode(
        variant.entry(),
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
            arm.state.tensor.binding(),
            arm.raw.tensor.binding(),
            arm.z.tensor.binding(),
            arm.norm_w.tensor.binding(),
            arm.gated.tensor.binding(),
        ],
        &[&(1.0f32 / 128.0f32.sqrt()).to_ne_bytes(), &1e-6f32.to_ne_bytes()],
        Grid::Threadgroups {
            groups: (H, 1, 1),
            threadgroup: (variant.threads(), 1, 1),
        },
    )
}

fn correctness(ctx: &MetalContext, variant: Variant, capture_raw: bool) -> Result<()> {
    for fixture in 0..3 {
        let baseline = Arm::new(ctx, fixture)?;
        let trial = Arm::new(ctx, fixture)?;
        for step in 0..STEPS {
            baseline.poison_outputs()?;
            trial.poison_outputs()?;
            let input = Inputs::new(ctx, fixture, step)?;
            let pass = ctx.begin_concurrent()?;
            reference(ctx, &pass, &baseline, &input)?;
            pass.level_barrier(&[
                &baseline.state.tensor,
                &baseline.window.tensor,
                &baseline.raw.tensor,
                &baseline.gated.tensor,
            ])?;
            fused(ctx, &pass, &trial, &input, variant, capture_raw)?;
            pass.commit()?.wait()?;
            baseline.check(true)?;
            trial.check(capture_raw)?;
            exact(&baseline.window.tensor, &trial.window.tensor, "history")?;
            exact(&baseline.conv.tensor, &trial.conv.tensor, "convolved QKV")?;
            exact(&baseline.state.tensor, &trial.state.tensor, "F32 recurrent state")?;
            if capture_raw {
                exact(&baseline.raw.tensor, &trial.raw.tensor, "BF16 raw output")?;
            }
            exact(&baseline.gated.tensor, &trial.gated.tensor, "BF16 gated output")?;
        }
    }
    println!(
        "gdn_fusion_exact variant={} capture_raw={} fixtures=3 steps_per_chain={} checks={}",
        variant.name(),
        capture_raw,
        STEPS,
        if capture_raw { 5 * 3 * STEPS } else { 4 * 3 * STEPS }
    );
    Ok(())
}

#[test]
#[ignore = "requires an explicitly available GPU slot; fused256 no-capture correctness"]
fn fused256_deployed_matches_production() -> Result<()> {
    correctness(&MetalContext::new()?, Variant::Fused256, false)
}

#[test]
#[ignore = "requires an explicitly available GPU slot; fused256 raw-capture correctness"]
fn fused256_capture_matches_production() -> Result<()> {
    correctness(&MetalContext::new()?, Variant::Fused256, true)
}

#[test]
#[ignore = "requires an explicitly available GPU slot; fused512 no-capture correctness"]
fn fused512_deployed_matches_production() -> Result<()> {
    correctness(&MetalContext::new()?, Variant::Fused512, false)
}

#[test]
#[ignore = "requires an explicitly available GPU slot; fused512 raw-capture correctness"]
fn fused512_capture_matches_production() -> Result<()> {
    correctness(&MetalContext::new()?, Variant::Fused512, true)
}

#[test]
#[ignore = "requires an explicitly available GPU slot; component timing only"]
fn fused_component_rotating_after_exactness() -> Result<()> {
    let ctx = MetalContext::new()?;
    correctness(&ctx, Variant::Fused256, false)?;
    correctness(&ctx, Variant::Fused512, false)?;
    let input = Inputs::new(&ctx, 1, 3)?;
    let arms = (0..36).map(|_| Arm::new(&ctx, 1)).collect::<Result<Vec<_>>>()?;
    let variants = [None, Some(Variant::Fused256), Some(Variant::Fused512)];
    let mut times = [Vec::<f64>::new(), Vec::new(), Vec::new()];
    let mut digests = [String::new(), String::new(), String::new()];
    for cycle in 0..4 {
        for arm_index in [0, 1, 2, 2, 1, 0] {
            for arm in &arms {
                arm.reset(1)?;
            }
            let pass = ctx.begin_concurrent()?;
            for arm in &arms {
                match variants[arm_index] {
                    None => reference(&ctx, &pass, arm, &input)?,
                    Some(v) => fused(&ctx, &pass, arm, &input, v, false)?,
                }
                pass.level_barrier(&[
                    &arm.window.tensor,
                    &arm.state.tensor,
                    &arm.raw.tensor,
                    &arm.gated.tensor,
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
                arm.check(arm_index == 0)?;
                identity.extend_from_slice(arm.state.tensor.raw_bytes());
                identity.extend_from_slice(arm.gated.tensor.raw_bytes());
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
    for (index, name) in ["production_conv_recurrence_normgate", "fused256", "fused512"]
        .iter()
        .enumerate()
    {
        let mean = times[index].iter().sum::<f64>() / times[index].len() as f64;
        println!(
            "gdn_fusion_component arm={name} samples={} mean_gpu_us={:.6} digest={}",
            times[index].len(),
            mean * 1e6,
            digests[index]
        );
        println!(
            "gdn_fusion_component_samples arm={name} gpu_seconds={:?}",
            times[index]
        );
    }
    Ok(())
}
