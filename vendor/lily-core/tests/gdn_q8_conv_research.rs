// SPDX-License-Identifier: Apache-2.0
//! Ignored exactness and streaming component controls for Q8 projection/conv.
//! Reference arrays stay in memory; output contains metrics, counts and hashes.
#![cfg(target_os = "macos")]

use anyhow::{Result, ensure};
use stark_lily_core::kernels::gdn::{GdnStateLayout, conv1d_step_with_layout};
use stark_lily_core::metal::{
    ComputePass, Grid, MetalContext, MslMathMode, MslVersion,
};
use stark_lily_core::sha256::{hex, sha256};
use stark_lily_core::tensor::{DType, Tensor};

const K: usize = 2560;
const N: usize = 16480;
const C: usize = 10240;
const GUARD: usize = 64;
const STEPS: usize = 16;
const LAYERS: usize = 36;
const STOCK: &str = include_str!("../src/kernels/metal/official_q8_gdn_decode.metal");
const FUSED: &str = include_str!(
    "../../../docs/archive/experiments/qwen38-gdn-next-2026-10-31/q8_conv.metal"
);

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

    fn guards(&self, category: &str) -> Result<()> {
        let raw = self.backing.raw_bytes();
        let edge = GUARD * self.tensor.dtype().size();
        ensure!(
            raw[..edge].iter().chain(&raw[raw.len() - edge..]).all(|x| *x == 0xa5),
            "{category}: canary changed"
        );
        Ok(())
    }

    fn finite(&self, category: &str) -> Result<()> {
        self.guards(category)?;
        ensure!(
            self.tensor.to_f32()?.iter().all(|x| x.is_finite()),
            "{category}: nonfinite result"
        );
        Ok(())
    }

    fn poison(&self) -> Result<()> {
        ensure!(self.tensor.dtype() == DType::BF16, "poison type");
        self.tensor.write_bytes(&[0xc1u8, 0x7f].repeat(self.tensor.numel()))
    }
}

fn bf_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_ne_bytes())
        .collect()
}

fn exact(a: &Tensor, b: &Tensor, category: &str) -> Result<()> {
    ensure!(a.shape() == b.shape() && a.dtype() == b.dtype(), "{category}: geometry");
    let changed = a
        .raw_bytes()
        .chunks_exact(a.dtype().size())
        .zip(b.raw_bytes().chunks_exact(b.dtype().size()))
        .filter(|(x, y)| x != y)
        .count();
    ensure!(changed == 0, "{category}: changed_elements={changed}");
    Ok(())
}

struct Inputs {
    codes: Guarded,
    scales: Guarded,
    biases: Guarded,
    x: Guarded,
    conv_w: Guarded,
}

impl Inputs {
    fn new(ctx: &MetalContext, fixture: usize, salt: usize) -> Result<Self> {
        let input = Self {
            codes: Guarded::new(ctx, &[N, K / 4], DType::U32)?,
            scales: Guarded::new(ctx, &[N, K / 64], DType::BF16)?,
            biases: Guarded::new(ctx, &[N, K / 64], DType::BF16)?,
            x: Guarded::new(ctx, &[K], DType::BF16)?,
            conv_w: Guarded::new(ctx, &[4, C], DType::BF16)?,
        };
        let mut codes = Vec::with_capacity(N * K / 4);
        for row in 0..N {
            for word in 0..K / 4 {
                let mut packed = 0u32;
                for lane in 0..4 {
                    let col = word * 4 + lane;
                    let value =
                        ((row * 7 + col * 3 + col / 64 + salt * 13 + fixture * 19)
                            % 256) as u32;
                    packed |= value << (8 * lane);
                }
                codes.push(packed);
            }
        }
        input.codes.tensor.write_bytes(bytemuck::cast_slice(&codes))?;
        let scales = (0..N)
            .flat_map(|row| {
                (0..K / 64).map(move |group| {
                    if fixture == 2 && (row + group) % 11 == 0 {
                        0.0
                    } else {
                        ((row * 11 + group * 5 + salt) % 23 + 1) as f32 / 1021.0
                    }
                })
            })
            .collect::<Vec<_>>();
        let biases = (0..N)
            .flat_map(|row| {
                (0..K / 64).map(move |group| {
                    (((row * 3 + group * 7 + salt * 11) % 19) as i32 - 9) as f32 / 251.0
                })
            })
            .collect::<Vec<_>>();
        input.scales.tensor.write_bytes(&bf_bytes(&scales))?;
        input.biases.tensor.write_bytes(&bf_bytes(&biases))?;
        let conv = (0..4 * C)
            .map(|i| {
                if fixture == 2 {
                    if i / C == 3 { 1.0 } else { 0.0 }
                } else {
                    (((i % C * 3 + i / C * 11 + salt) % 31) as i32 - 15) as f32 / 37.0
                }
            })
            .collect::<Vec<_>>();
        input.conv_w.tensor.write_bytes(&bf_bytes(&conv))?;
        input.set_step(fixture, 0)?;
        Ok(input)
    }

    fn set_step(&self, fixture: usize, step: usize) -> Result<()> {
        const EDGE: [u16; 12] = [
            0x0000, 0x8000, 0x3f80, 0xbf80, 0x4180, 0xc180, 0x42a0, 0xc2a0, 0x3b80,
            0xbb80, 0x3eab, 0xbeab,
        ];
        let values = (0..K)
            .map(|i| {
                if fixture == 2 {
                    half::bf16::from_bits(EDGE[(i + step) % EDGE.len()]).to_f32()
                } else {
                    (((i * 17 + step * 7 + fixture * 11) % 71) as i32 - 35) as f32
                        / 23.0
                }
            })
            .collect::<Vec<_>>();
        self.x.tensor.write_bytes(&bf_bytes(&values))
    }

    fn fingerprint(&self) -> Vec<String> {
        [&self.codes, &self.scales, &self.biases, &self.conv_w]
            .iter()
            .map(|x| hex(&sha256(x.tensor.raw_bytes())))
            .collect()
    }

    fn check(&self) -> Result<()> {
        self.codes.guards("codes")?;
        self.scales.finite("scales")?;
        self.biases.finite("biases")?;
        self.x.finite("input")?;
        self.conv_w.finite("conv weights")
    }
}

struct Arm {
    history: Guarded,
    joined: Guarded,
    convolved: Guarded,
}

impl Arm {
    fn new(ctx: &MetalContext, fixture: usize) -> Result<Self> {
        let arm = Self {
            history: Guarded::new(ctx, &[C, 3], DType::BF16)?,
            joined: Guarded::new(ctx, &[N], DType::BF16)?,
            convolved: Guarded::new(ctx, &[C], DType::BF16)?,
        };
        arm.reset(fixture)?;
        Ok(arm)
    }

    fn reset(&self, fixture: usize) -> Result<()> {
        let history = (0..C * 3)
            .map(|i| {
                if fixture == 0 {
                    0.0
                } else {
                    (((i * 7 + fixture * 3) % 59) as i32 - 29) as f32 / 47.0
                }
            })
            .collect::<Vec<_>>();
        self.history.tensor.write_bytes(&bf_bytes(&history))?;
        self.poison_outputs()
    }

    fn poison_outputs(&self) -> Result<()> {
        self.joined.poison()?;
        self.convolved.poison()
    }

    fn check(&self, capture_raw: bool) -> Result<()> {
        self.history.finite("history")?;
        self.convolved.finite("convolved")?;
        self.joined.guards("joined")?;
        if capture_raw {
            self.joined.finite("joined")?;
        } else {
            let raw = self.joined.tensor.view(0, &[C])?;
            ensure!(
                raw.raw_bytes() == [0xc1u8, 0x7f].repeat(C),
                "unused raw QKV changed"
            );
            ensure!(
                self.joined
                    .tensor
                    .view(C, &[N - C])?
                    .to_f32()?
                    .iter()
                    .all(|x| x.is_finite()),
                "joined tail nonfinite"
            );
        }
        Ok(())
    }
}

fn reference(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    input: &Inputs,
    arm: &Arm,
) -> Result<()> {
    let pipeline = ctx.pipeline_with_math_mode(
        "official_q8_gdn_decode_qmv_fast_g64_bf16",
        STOCK,
        MslVersion::V4_0,
        MslMathMode::Safe,
    )?;
    ensure!(pipeline.thread_execution_width() == 32, "reference SIMD width");
    pass.dispatch_at(
        &pipeline,
        &[
            input.codes.tensor.binding(),
            input.scales.tensor.binding(),
            input.biases.tensor.binding(),
            input.x.tensor.binding(),
            arm.joined.tensor.binding(),
        ],
        &[&(K as i32).to_ne_bytes(), &(N as i32).to_ne_bytes()],
        Grid::Threadgroups { groups: (1, N / 8, 1), threadgroup: (32, 2, 1) },
    )?;
    pass.level_barrier(&[&arm.joined.tensor])?;
    conv1d_step_with_layout(
        ctx,
        pass,
        &arm.history.tensor,
        &arm.joined.tensor.view(0, &[C])?,
        &input.conv_w.tensor,
        &arm.convolved.tensor,
        true,
        GdnStateLayout::ValueKey,
    )
}

fn candidate(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    input: &Inputs,
    arm: &Arm,
    capture_raw: bool,
) -> Result<()> {
    let source = if capture_raw {
        format!("#define RESEARCH_CAPTURE_RAW_QKV 1\n{FUSED}")
    } else {
        FUSED.to_owned()
    };
    let pipeline = ctx.pipeline_with_math_mode(
        "qwen38_q8_projection_conv_research",
        &source,
        MslVersion::V4_0,
        MslMathMode::Safe,
    )?;
    ensure!(pipeline.thread_execution_width() == 32, "candidate SIMD width");
    pass.dispatch_at(
        &pipeline,
        &[
            input.codes.tensor.binding(),
            input.scales.tensor.binding(),
            input.biases.tensor.binding(),
            input.x.tensor.binding(),
            arm.joined.tensor.binding(),
            arm.history.tensor.binding(),
            input.conv_w.tensor.binding(),
            arm.convolved.tensor.binding(),
        ],
        &[&(K as i32).to_ne_bytes(), &(N as i32).to_ne_bytes()],
        Grid::Threadgroups { groups: (1, N / 8, 1), threadgroup: (32, 2, 1) },
    )
}

fn correctness(ctx: &MetalContext, capture_raw: bool) -> Result<()> {
    for fixture in 0..3 {
        let input = Inputs::new(ctx, fixture, 0)?;
        let weights_before = input.fingerprint();
        let baseline = Arm::new(ctx, fixture)?;
        let trial = Arm::new(ctx, fixture)?;
        for step in 0..STEPS {
            input.set_step(fixture, step)?;
            let x_before = input.x.tensor.raw_bytes().to_vec();
            baseline.poison_outputs()?;
            trial.poison_outputs()?;
            let pass = ctx.begin_concurrent()?;
            reference(ctx, &pass, &input, &baseline)?;
            pass.level_barrier(&[
                &baseline.history.tensor,
                &baseline.joined.tensor,
                &baseline.convolved.tensor,
            ])?;
            candidate(ctx, &pass, &input, &trial, capture_raw)?;
            pass.commit()?.wait()?;
            baseline.check(true)?;
            trial.check(capture_raw)?;
            input.check()?;
            ensure!(input.x.tensor.raw_bytes() == x_before, "input changed");
            exact(&baseline.history.tensor, &trial.history.tensor, "history")?;
            exact(
                &baseline.convolved.tensor,
                &trial.convolved.tensor,
                "convolved QKV",
            )?;
            if capture_raw {
                exact(
                    &baseline.joined.tensor,
                    &trial.joined.tensor,
                    "full projected joined output",
                )?;
            } else {
                exact(
                    &baseline.joined.tensor.view(C, &[N - C])?,
                    &trial.joined.tensor.view(C, &[N - C])?,
                    "projected z/a/b",
                )?;
            }
        }
        ensure!(
            input.fingerprint() == weights_before,
            "immutable projection/conv weights changed"
        );
    }
    println!(
        "q8_conv_exact capture_raw={capture_raw} fixtures=3 steps_per_chain={STEPS} whole_array_checks={}",
        3 * 3 * STEPS
    );
    Ok(())
}

#[test]
#[ignore = "requires an available GPU slot; projection-conv deployed control"]
fn q8_conv_deployed_matches_production() -> Result<()> {
    correctness(&MetalContext::new()?, false)
}

#[test]
#[ignore = "requires an available GPU slot; diagnostic raw-projection control"]
fn q8_conv_capture_matches_production() -> Result<()> {
    correctness(&MetalContext::new()?, true)
}

#[test]
#[ignore = "requires an available GPU slot; component timing only"]
fn q8_conv_component_streaming_after_exactness() -> Result<()> {
    let ctx = MetalContext::new()?;
    correctness(&ctx, false)?;
    // MetalContext caches by entry name/version/math mode, not source text.
    // Keep the macro-enabled capture kernel out of the deployed timing cache.
    correctness(&MetalContext::new()?, true)?;
    // Distinct actual-shape weight buffers per layer avoid measuring a single
    // repeatedly cached 45MB matrix instead of the 36-layer GDN stream.
    let inputs =
        (0..LAYERS).map(|i| Inputs::new(&ctx, 1, i)).collect::<Result<Vec<_>>>()?;
    let input_hashes = inputs.iter().map(Inputs::fingerprint).collect::<Vec<_>>();
    let arms = (0..LAYERS).map(|_| Arm::new(&ctx, 1)).collect::<Result<Vec<_>>>()?;
    let mut times = [Vec::<f64>::new(), Vec::new()];
    let mut digests = [String::new(), String::new()];
    for cycle in 0..4 {
        for arm_index in [0, 1, 1, 0] {
            for arm in &arms {
                arm.reset(1)?;
            }
            let pass = ctx.begin_concurrent()?;
            for (input, arm) in inputs.iter().zip(&arms) {
                if arm_index == 0 {
                    reference(&ctx, &pass, input, arm)?;
                } else {
                    candidate(&ctx, &pass, input, arm, false)?;
                }
                pass.level_barrier(&[
                    &arm.history.tensor,
                    &arm.joined.tensor,
                    &arm.convolved.tensor,
                ])?;
            }
            let completed = pass.commit()?.wait_retain()?;
            let timing = completed.timing()?;
            let seconds = timing.gpu_end_secs - timing.gpu_start_secs;
            ensure!(seconds.is_finite() && seconds > 0.0, "invalid component timing");
            if cycle > 0 {
                times[arm_index].push(seconds / LAYERS as f64);
            }
            let mut identity = Vec::new();
            for arm in &arms {
                arm.check(arm_index == 0)?;
                identity.extend_from_slice(arm.history.tensor.raw_bytes());
                identity.extend_from_slice(arm.convolved.tensor.raw_bytes());
                identity.extend_from_slice(
                    arm.joined.tensor.view(C, &[N - C])?.raw_bytes(),
                );
            }
            let digest = hex(&sha256(&identity));
            if digests[arm_index].is_empty() {
                digests[arm_index] = digest;
            } else {
                ensure!(digests[arm_index] == digest, "timed repeat digest changed");
            }
        }
    }
    ensure!(digests[0] == digests[1], "timed cross-arm digest differs");
    for (input, original) in inputs.iter().zip(input_hashes) {
        input.check()?;
        ensure!(input.fingerprint() == original, "timed immutable weights changed");
    }
    for (index, name) in
        ["stock_projection_then_conv", "q8_projection_conv_fused"].iter().enumerate()
    {
        let mean = times[index].iter().sum::<f64>() / times[index].len() as f64;
        println!(
            "q8_conv_component arm={name} samples={} mean_gpu_us={:.6} digest={}",
            times[index].len(),
            mean * 1e6,
            digests[index]
        );
        println!("q8_conv_component_samples arm={name} gpu_seconds={:?}", times[index]);
    }
    Ok(())
}
