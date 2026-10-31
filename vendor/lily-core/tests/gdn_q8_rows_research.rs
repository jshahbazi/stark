// SPDX-License-Identifier: Apache-2.0
//! Opt-in exact controls and component measurements for Q8 input-row reuse.
//! No checkpoint is read. Only aggregate counts, hashes and timings are printed.
#![cfg(target_os = "macos")]

use anyhow::{Result, ensure};
use stark_lily_core::metal::{
    ComputePass, Grid, MetalContext, MslMathMode, MslVersion,
};
use stark_lily_core::sha256::{hex, sha256};
use stark_lily_core::tensor::{DType, Tensor};

const STOCK: &str = include_str!("../src/kernels/metal/official_q8_gdn_decode.metal");
const REUSE: &str = include_str!(
    "../../../docs/archive/experiments/qwen38-gdn-next-2026-10-31/q8_rows.metal"
);
const GUARD: usize = 64;
const STEPS: usize = 4;
const SEED: usize = 0x384864;
const SHAPES: [(usize, usize); 2] = [(2560, 16480), (6144, 2560)];
// Existing admitted stock controls at M1, identical fixture0/step0 values.
// These bind the unchanged reference rather than silently manufacturing a
// candidate golden. All other rows/fixtures compare directly to that source.
const STOCK_HASHES: [&str; 2] = [
    "ffe2bff0fe897d2b7f4ed08d4b5b8004c8c65b5d61a1f3d3305ba2d42028d459",
    "c4026ed6d8fa41c2151df73d60b377dd125e820d69d2da8af53d1637127796fc",
];

fn digest(tensor: &Tensor) -> String {
    hex(&sha256(tensor.raw_bytes()))
}

fn bf_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_ne_bytes())
        .collect()
}

struct Guarded {
    backing: Tensor,
    tensor: Tensor,
}

impl Guarded {
    fn new(
        ctx: &MetalContext,
        shape: &[usize],
        dtype: DType,
        raw: &[u8],
    ) -> Result<Self> {
        let count = shape.iter().product::<usize>();
        ensure!(raw.len() == count * dtype.size(), "fixture byte geometry");
        let mut bytes = vec![0xa5; GUARD * dtype.size()];
        bytes.extend_from_slice(raw);
        bytes.extend_from_slice(&vec![0xa5; GUARD * dtype.size()]);
        let backing = Tensor::from_bytes(ctx, &bytes, &[count + 2 * GUARD], dtype)?;
        let tensor = backing.view(GUARD, shape)?;
        Ok(Self { backing, tensor })
    }

    fn bf16(ctx: &MetalContext, shape: &[usize]) -> Result<Self> {
        Self::new(
            ctx,
            shape,
            DType::BF16,
            &vec![0; shape.iter().product::<usize>() * 2],
        )
    }

    fn check_guards(&self, category: &str) -> Result<()> {
        let raw = self.backing.raw_bytes();
        let edge = GUARD * self.tensor.dtype().size();
        ensure!(
            raw[..edge].iter().chain(&raw[raw.len() - edge..]).all(|b| *b == 0xa5),
            "{category}: canary changed"
        );
        Ok(())
    }

    fn check_finite(&self, category: &str) -> Result<()> {
        self.check_guards(category)?;
        ensure!(
            self.tensor.to_f32()?.iter().all(|v| v.is_finite()),
            "{category}: nonfinite value"
        );
        Ok(())
    }

    fn poison(&self) -> Result<()> {
        self.tensor.write_bytes(&[0xc1u8, 0x7f].repeat(self.tensor.numel()))
    }
}

struct Fixture {
    k: usize,
    n: usize,
    codes: Guarded,
    scales: Guarded,
    biases: Guarded,
    x: Guarded,
    weight_hashes: [String; 3],
}

impl Fixture {
    fn new(
        ctx: &MetalContext,
        k: usize,
        n: usize,
        kind: usize,
        salt: usize,
    ) -> Result<Self> {
        let seed = SEED + salt * 991;
        let mut codes = Vec::with_capacity(n * k / 4);
        for row in 0..n {
            for word in 0..k / 4 {
                let mut packed = 0u32;
                for lane in 0..4 {
                    let col = word * 4 + lane;
                    let code = match kind {
                        0 => (row * 7 + col * 3 + col / 64 + seed) % 256,
                        1 => {
                            let key = (row as u64 * 0x9e3779b1)
                                ^ (col as u64 * 0x85ebca77)
                                ^ seed as u64;
                            ((key ^ (key >> 13) ^ (key >> 23)) & 255) as usize
                        }
                        _ => [0, 1, 127, 128, 254, 255][(row * 3 + col * 5 + seed) % 6],
                    };
                    packed |= (code as u32) << (lane * 8);
                }
                codes.push(packed);
            }
        }
        let mut scales = Vec::with_capacity(n * k / 64);
        let mut biases = Vec::with_capacity(n * k / 64);
        for row in 0..n {
            for group in 0..k / 64 {
                scales.push(match kind {
                    0 => ((row * 11 + group * 5 + seed) % 23 + 1) as f32 / 512.0,
                    1 => {
                        (((row * 13 + group * 17 + seed) % 43) as i32 - 21) as f32
                            / 1536.0
                    }
                    _ => [0.0, -0.0, 0.0009765625, -0.0009765625, 0.003891, -0.003891]
                        [(row * 7 + group + seed) % 6],
                });
                biases.push(match kind {
                    0 => {
                        (((row * 3 + group * 7 + seed) % 19) as i32 - 9) as f32 / 256.0
                    }
                    1 => {
                        (((row * 5 + group * 11 + seed) % 37) as i32 - 18) as f32
                            / 127.0
                    }
                    _ => [
                        0.0,
                        -0.0,
                        0.12451171875,
                        -0.12451171875,
                        0.333251953125,
                        -0.333251953125,
                    ][(row * 3 + group + seed) % 6],
                });
            }
        }
        let codes =
            Guarded::new(ctx, &[n, k / 4], DType::U32, bytemuck::cast_slice(&codes))?;
        let scales = Guarded::new(ctx, &[n, k / 64], DType::BF16, &bf_bytes(&scales))?;
        let biases = Guarded::new(ctx, &[n, k / 64], DType::BF16, &bf_bytes(&biases))?;
        let weight_hashes =
            [digest(&codes.tensor), digest(&scales.tensor), digest(&biases.tensor)];
        Ok(Self {
            k,
            n,
            codes,
            scales,
            biases,
            x: Guarded::bf16(ctx, &[4, k])?,
            weight_hashes,
        })
    }

    fn set_input(
        &self,
        rows: usize,
        kind: usize,
        step: usize,
        salt: usize,
    ) -> Result<(Tensor, String)> {
        self.x.poison()?;
        let seed = SEED + salt * 991;
        let values = (0..rows * self.k)
            .map(|index| {
                let row = index / self.k;
                let col = index % self.k;
                match kind {
                    0 => {
                        (((col * 17 + seed + row * 29 + step * 13) % 71) as i32 - 35)
                            as f32
                            / 23.0
                    }
                    1 => {
                        (((col * 13 + row * 19 + step * 7 + seed) % 113) as i32 - 56)
                            as f32
                            / 47.0
                            + (row + step) as f32 / 257.0
                    }
                    _ => {
                        const EDGES: [u16; 16] = [
                            0x0000, 0x8000, 0x0001, 0x8001, 0x3eab, 0xbeab, 0x3f01,
                            0xbf01, 0x3f7f, 0xbf7f, 0x3f81, 0xbf81, 0x411f, 0xc11f,
                            0x42a0, 0xc2a0,
                        ];
                        half::bf16::from_bits(
                            EDGES[(col * 7 + row * 11 + step + salt) % EDGES.len()],
                        )
                        .to_f32()
                    }
                }
            })
            .collect::<Vec<_>>();
        let view = self.x.tensor.view(0, &[rows, self.k])?;
        view.write_bytes(&bf_bytes(&values))?;
        ensure!(view.to_f32()?.iter().all(|v| v.is_finite()), "input nonfinite");
        Ok((view, digest(&self.x.tensor)))
    }

    fn check_guards(&self) -> Result<()> {
        self.codes.check_guards("codes")?;
        self.scales.check_finite("scales")?;
        self.biases.check_finite("biases")?;
        self.x.check_guards("input")
    }

    fn check_unchanged(&self, input_hash: &str) -> Result<()> {
        self.check_guards()?;
        ensure!(
            [
                digest(&self.codes.tensor),
                digest(&self.scales.tensor),
                digest(&self.biases.tensor)
            ] == self.weight_hashes,
            "weight arrays changed"
        );
        ensure!(digest(&self.x.tensor) == input_hash, "input array changed");
        Ok(())
    }
}

fn dispatch(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    fixture: &Fixture,
    x: &Tensor,
    output: &Tensor,
    rows: usize,
    candidate: bool,
) -> Result<()> {
    ensure!(
        matches!((fixture.k, fixture.n), (2560, 16480) | (6144, 2560)),
        "Q8 research shape"
    );
    ensure!(
        (1..=4).contains(&rows)
            && x.numel() == rows * fixture.k
            && output.numel() == rows * fixture.n,
        "Q8 research rows"
    );
    let entry = match rows {
        1 => "qwen38_q8_gdn_rows_1_research",
        2 => "qwen38_q8_gdn_rows_2_research",
        3 => "qwen38_q8_gdn_rows_3_research",
        4 => "qwen38_q8_gdn_rows_4_research",
        _ => unreachable!(),
    };
    let pipeline = ctx.pipeline_with_math_mode(
        if candidate { entry } else { "official_q8_gdn_decode_qmv_fast_g64_bf16" },
        if candidate { REUSE } else { STOCK },
        MslVersion::V4_0,
        MslMathMode::Safe,
    )?;
    ensure!(pipeline.thread_execution_width() == 32, "Q8 research requires SIMD32");
    pass.dispatch_at(
        &pipeline,
        &[
            fixture.codes.tensor.binding(),
            fixture.scales.tensor.binding(),
            fixture.biases.tensor.binding(),
            x.binding(),
            output.binding(),
        ],
        &[&(fixture.k as i32).to_ne_bytes(), &(fixture.n as i32).to_ne_bytes()],
        Grid::Threadgroups {
            groups: (if candidate { 1 } else { rows }, fixture.n / 8, 1),
            threadgroup: (32, 2, 1),
        },
    )
}

fn exact(reference: &Tensor, candidate: &Tensor, category: &str) -> Result<()> {
    ensure!(
        reference.shape() == candidate.shape()
            && reference.dtype() == candidate.dtype(),
        "{category}: geometry"
    );
    let mismatches = reference
        .raw_bytes()
        .chunks_exact(2)
        .zip(candidate.raw_bytes().chunks_exact(2))
        .filter(|(a, b)| a != b)
        .count();
    ensure!(mismatches == 0, "{category}: mismatched_elements={mismatches}");
    Ok(())
}

fn correctness(ctx: &MetalContext, rows: usize) -> Result<()> {
    for (geometry, (k, n)) in SHAPES.into_iter().enumerate() {
        for kind in 0..3 {
            let fixture = Fixture::new(ctx, k, n, kind, 0)?;
            let reference = Guarded::bf16(ctx, &[rows, n])?;
            let trial = Guarded::bf16(ctx, &[rows, n])?;
            let repeat = Guarded::bf16(ctx, &[rows, n])?;
            for step in 0..STEPS {
                let (x, input_hash) = fixture.set_input(rows, kind, step, 0)?;
                reference.poison()?;
                trial.poison()?;
                repeat.poison()?;
                let pass = ctx.begin_concurrent()?;
                dispatch(ctx, &pass, &fixture, &x, &reference.tensor, rows, false)?;
                pass.level_barrier(&[&reference.tensor])?;
                dispatch(ctx, &pass, &fixture, &x, &trial.tensor, rows, true)?;
                pass.level_barrier(&[&trial.tensor])?;
                dispatch(ctx, &pass, &fixture, &x, &repeat.tensor, rows, true)?;
                pass.commit()?.wait()?;
                fixture.check_unchanged(&input_hash)?;
                reference.check_finite("reference")?;
                trial.check_finite("candidate")?;
                repeat.check_finite("repeat candidate")?;
                exact(&reference.tensor, &trial.tensor, "stock vs reused BF16 output")?;
                exact(&trial.tensor, &repeat.tensor, "reused repeat BF16 output")?;
                if rows == 1 && kind == 0 && step == 0 {
                    ensure!(
                        digest(&reference.tensor) == STOCK_HASHES[geometry],
                        "unchanged reference stock golden mismatch"
                    );
                }
            }
        }
    }
    println!(
        "gdn_q8_rows_exact rows={rows} shapes=2 fixtures=3 input_steps={STEPS} output_array_comparisons={} unchanged_inputs=true poison_canaries_finiteness=true",
        2 * 3 * STEPS * 2
    );
    Ok(())
}

#[test]
#[ignore = "requires explicitly available GPU; diagnostic M1 control only"]
fn q8_rows1_matches_stock_exactly() -> Result<()> {
    correctness(&MetalContext::new()?, 1)
}

#[test]
#[ignore = "requires explicitly available GPU; M2 reuse exactness"]
fn q8_rows2_matches_stock_exactly() -> Result<()> {
    correctness(&MetalContext::new()?, 2)
}

#[test]
#[ignore = "requires explicitly available GPU; M3 reuse exactness"]
fn q8_rows3_matches_stock_exactly() -> Result<()> {
    correctness(&MetalContext::new()?, 3)
}

#[test]
#[ignore = "requires explicitly available GPU; M4 reuse exactness"]
fn q8_rows4_matches_stock_exactly() -> Result<()> {
    correctness(&MetalContext::new()?, 4)
}

#[test]
#[ignore = "requires explicitly available GPU; rotating component controls"]
fn q8_rows_component_rotating_after_exactness() -> Result<()> {
    let ctx = MetalContext::new()?;
    for rows in 1..=4 {
        correctness(&ctx, rows)?;
    }
    // Nine independent weight sets avoid measuring only one repeated matrix.
    // This remains synthetic: full-model data/scheduling must be measured.
    for (k, n) in SHAPES {
        let fixtures = (0..9)
            .map(|salt| Fixture::new(&ctx, k, n, 1, salt))
            .collect::<Result<Vec<_>>>()?;
        for rows in 2..=4 {
            let inputs = fixtures
                .iter()
                .enumerate()
                .map(|(salt, f)| f.set_input(rows, 1, 3, salt))
                .collect::<Result<Vec<_>>>()?;
            let outputs = (0..9)
                .map(|_| Guarded::bf16(&ctx, &[rows, n]))
                .collect::<Result<Vec<_>>>()?;
            let mut times = [Vec::<f64>::new(), Vec::<f64>::new()];
            let mut expected_digest = String::new();
            for cycle in 0..4 {
                // Alternate orientation across cycles; excluded warm cycle0.
                let order = if cycle % 2 == 0 { [0, 1, 1, 0] } else { [1, 0, 0, 1] };
                for (position, arm) in order.into_iter().enumerate() {
                    for output in &outputs {
                        output.poison()?;
                    }
                    let pass = ctx.begin_concurrent()?;
                    for repeat in 0..4 {
                        for offset in 0..9 {
                            let index = (offset + cycle + repeat) % 9;
                            dispatch(
                                &ctx,
                                &pass,
                                &fixtures[index],
                                &inputs[index].0,
                                &outputs[index].tensor,
                                rows,
                                arm == 1,
                            )?;
                            pass.level_barrier(&[&outputs[index].tensor])?;
                        }
                    }
                    let completed = pass.commit()?.wait_retain()?;
                    let timing = completed.timing()?;
                    let seconds = timing.gpu_end_secs - timing.gpu_start_secs;
                    ensure!(
                        seconds.is_finite() && seconds > 0.0,
                        "invalid Q8 component timing"
                    );
                    if cycle > 0 {
                        times[arm].push(seconds / 36.0);
                    }
                    let mut identity = Vec::new();
                    for output in &outputs {
                        output.check_finite("timed output")?;
                        identity.extend_from_slice(output.tensor.raw_bytes());
                    }
                    let hash = hex(&sha256(&identity));
                    if expected_digest.is_empty() {
                        expected_digest = hash;
                    } else {
                        ensure!(
                            expected_digest == hash,
                            "timed repeated/cross-arm digest mismatch"
                        );
                    }
                    println!(
                        "gdn_q8_rows_component_sample k={k} n={n} rows={rows} cycle={cycle} position={position} arm={} warmup={} gpu_seconds_per_projection={:.12}",
                        if arm == 0 { "stock" } else { "reuse" },
                        cycle == 0,
                        seconds / 36.0
                    );
                }
            }
            for (fixture, (_, input_hash)) in fixtures.iter().zip(&inputs) {
                fixture.check_unchanged(input_hash)?;
            }
            for (arm, name) in ["stock", "reuse"].iter().enumerate() {
                let mean = times[arm].iter().sum::<f64>() / times[arm].len() as f64;
                println!(
                    "gdn_q8_rows_component k={k} n={n} rows={rows} arm={name} samples={} mean_gpu_us_per_projection={:.6} mean_gpu_us_per_input_row={:.6} weight_sets=9 serialized_projections=36 digest={expected_digest}",
                    times[arm].len(),
                    mean * 1e6,
                    mean * 1e6 / rows as f64
                );
                println!(
                    "gdn_q8_rows_component_samples k={k} n={n} rows={rows} arm={name} gpu_seconds_per_projection={:?}",
                    times[arm]
                );
            }
        }
    }
    Ok(())
}
