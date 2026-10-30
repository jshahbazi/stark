//! Branch-matched projection controls for the prefill precision policy.
//!
//! These tests do not set an end-to-end model quality tolerance. They isolate
//! the standard GEMV affine arithmetic from GEMM's BF16 weight staging and
//! compare each branch with an independent host F64 dot-product oracle.
use half::bf16;
use stark_engine_macros::stark_test;

use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, Kernels,
            gpu_types::QuantizationMode,
            kernel::matmul::{
                MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, QuantParams, QuantParamsLayout,
                QuantizedB, QuantizedCorrection,
            },
        },
        metal::{GemmEngine, Metal},
    },
    data_type::DataType,
};

type MetalContext = <Metal as Backend>::Context;
type MetalBuffer = <Metal as Backend>::GlobalBuffer;
type MatrixKernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;

const N: usize = 68;
const K: usize = 256;
const GROUP: usize = 64;
const MAX_ROWS: usize = 32;

struct Projection {
    bits: u32,
    input: Vec<bf16>,
    affine: Vec<f64>,
    staged: Vec<bf16>,
    factor_magnitudes: Vec<f64>,
    input_buffer: MetalBuffer,
    codes: MetalBuffer,
    scales: MetalBuffer,
    biases: MetalBuffer,
    dense: MetalBuffer,
}

fn upload(
    context: &MetalContext,
    bytes: &[u8],
) -> MetalBuffer {
    let mut buffer =
        context.create_buffer(bytes.len()).unwrap_or_else(|_| panic!("projection buffer allocation failed"));
    (&mut buffer).as_slice_mut::<u8>().copy_from_slice(bytes);
    buffer
}

impl Projection {
    fn cancellation(
        context: &MetalContext,
        bits: u32,
    ) -> Self {
        let mut projection = Self::new(context, bits, true);
        let origin = 1_u8 << (bits - 1);
        let mut codes = vec![
            if bits == 4 {
                origin * 0x11
            } else {
                origin
            };
            N * K * bits as usize / 8
        ];
        projection.input.fill(bf16::ZERO);
        projection.affine.fill(0.0);
        projection.staged.fill(bf16::ZERO);
        projection.factor_magnitudes.fill(2.0 * f64::from(origin));
        for row in 0..MAX_ROWS {
            projection.input[row * K] = bf16::ONE;
            projection.input[row * K + 1] = bf16::from_f32(1.0 / 512.0);
            projection.input[row * K + K / 2] = bf16::ONE;
        }
        for row in 0..N {
            let sign = if row.is_multiple_of(2) {
                1_i16
            } else {
                -1_i16
            };
            for (inner, value) in [(0, sign), (1, sign), (K / 2, -sign)] {
                let index = row * K + inner;
                let code = (i16::from(origin) + value) as u8;
                if bits == 4 {
                    let shift = 4 * (index % 2);
                    codes[index / 2] = (codes[index / 2] & !(0x0f << shift)) | (code << shift);
                } else {
                    codes[index] = code;
                }
                projection.affine[index] = f64::from(value);
                projection.staged[index] = bf16::from_f32(f32::from(value));
                projection.factor_magnitudes[index] = f64::from(code) + f64::from(origin);
            }
        }
        let scales = vec![bf16::ONE; K / GROUP * N];
        let biases = vec![bf16::from_f32(-f32::from(origin)); K / GROUP * N];
        projection.input_buffer = upload(context, bytemuck::cast_slice(&projection.input));
        projection.codes = upload(context, &codes);
        projection.scales = upload(context, bytemuck::cast_slice(&scales));
        projection.biases = upload(context, bytemuck::cast_slice(&biases));
        projection.dense = upload(context, bytemuck::cast_slice(&projection.staged));
        projection
    }

    fn new(
        context: &MetalContext,
        bits: u32,
        exact_affines: bool,
    ) -> Self {
        let input = (0..MAX_ROWS * K)
            .map(|i| bf16::from_f32(((i * 13 + i / K * 7) % 33) as f32 / 32.0 - 0.5))
            .collect::<Vec<_>>();
        let groups = K / GROUP;
        let mut scales = vec![bf16::ZERO; groups * N];
        let mut biases = vec![bf16::ZERO; groups * N];
        for group in 0..groups {
            for row in 0..N {
                let scale = 2.0_f32.powi(-12 + ((row + group * 3) % 5) as i32);
                let origin = if exact_affines {
                    (1_u32 << (bits - 1)) as f32
                } else {
                    // BF16-exact bias and power-of-two scale. Scale*code is
                    // exactly BF16, so only the final affine addition rounds.
                    // Its asymmetric origin makes many resulting weights
                    // require more than BF16's eight significant bits.
                    if (row + group).is_multiple_of(2) {
                        129.0 / 128.0
                    } else {
                        -129.0 / 128.0
                    }
                };
                scales[group * N + row] = bf16::from_f32(scale);
                biases[group * N + row] = bf16::from_f32(-origin * scale);
            }
        }
        let mut codes = vec![0_u8; N * K * bits as usize / 8];
        let mut affine = Vec::with_capacity(N * K);
        let mut staged = Vec::with_capacity(N * K);
        let mut factor_magnitudes = Vec::with_capacity(N * K);
        for row in 0..N {
            for inner in 0..K {
                let code = ((row * 37 + inner * 13 + inner / GROUP * 7) & ((1 << bits) - 1)) as u8;
                let index = row * K + inner;
                if bits == 4 {
                    codes[index / 2] |= code << (4 * (index % 2));
                } else {
                    codes[index] = code;
                }
                let parameter = inner / GROUP * N + row;
                let scale = f64::from(scales[parameter].to_f32());
                let bias = f64::from(biases[parameter].to_f32());
                let value = scale * f64::from(code) + bias;
                assert!(f64::from(value as f32) == value, "oracle affine value exceeded exact F32 construction");
                affine.push(value);
                staged.push(bf16::from_f32(value as f32));
                factor_magnitudes.push((scale * f64::from(code)).abs() + bias.abs());
            }
        }
        let has_staging_rounding = affine.iter().zip(&staged).any(|(&a, b)| a != f64::from(b.to_f32()));
        assert!(has_staging_rounding != exact_affines, "projection fixture did not separate affine policies");
        Self {
            bits,
            input_buffer: upload(context, bytemuck::cast_slice(&input)),
            codes: upload(context, &codes),
            scales: upload(context, bytemuck::cast_slice(&scales)),
            biases: upload(context, bytemuck::cast_slice(&biases)),
            dense: upload(context, bytemuck::cast_slice(&staged)),
            input,
            affine,
            staged,
            factor_magnitudes,
        }
    }

    fn quantized(&self) -> MatmulB<&MetalBuffer> {
        MatmulB::Quantized(QuantizedB {
            codes: &self.codes,
            scales: &self.scales,
            correction: QuantizedCorrection::Biases(&self.biases),
            params: QuantParams::new(QuantParamsLayout::GroupOutput, N as u32, (K / GROUP) as u32),
            mode: if self.bits == 4 {
                QuantizationMode::U4
            } else {
                QuantizationMode::U8
            },
            group_size: GROUP as u32,
            signed_codes: false,
        })
    }
}

#[derive(Clone, Copy)]
enum Arm {
    QuantizedGemv,
    QuantizedGemm,
    QuantizedUnsplitGemm,
    QuantizedSplitGemm(u32),
    DenseStagedGemm,
}

fn execute(
    context: &MetalContext,
    projection: &Projection,
    rows: usize,
    dtype: DataType,
    arm: Arm,
) -> Vec<f64> {
    let mut kernel = MatrixKernel::new(context, DataType::BF16, DataType::BF16, dtype)
        .unwrap_or_else(|_| panic!("projection kernel construction failed"));
    execute_with_kernel(context, projection, rows, dtype, arm, &mut kernel)
}

fn execute_with_kernel(
    context: &MetalContext,
    projection: &Projection,
    rows: usize,
    dtype: DataType,
    arm: Arm,
    kernel: &mut MatrixKernel,
) -> Vec<f64> {
    let mut commands =
        context.create_command_buffer(None, None).unwrap_or_else(|_| panic!("projection command allocation failed"));
    let mut output = commands
        .allocate_scratch_for_shape(&[rows as u32, N as u32], dtype)
        .unwrap_or_else(|_| panic!("projection output allocation failed"));
    let calls = if matches!(arm, Arm::QuantizedGemv) {
        rows
    } else {
        1
    };
    let call_rows = rows / calls;
    for call in 0..calls {
        let first = call * call_rows;
        let b = match arm {
            Arm::DenseStagedGemm => MatmulB::FullPrecision {
                b: &projection.dense,
            },
            _ => projection.quantized(),
        };
        let arguments = MatmulArguments {
            a: MatmulA::FullPrecision {
                values: (&projection.input_buffer).subrange(first * K * 2..(first + call_rows) * K * 2),
                offset: 0,
            },
            b,
            b_leading_dimension: None,
            b_transpose: true,
            d: (&mut output)
                .subrange_mut(first * N * dtype.size_in_bytes()..(first + call_rows) * N * dtype.size_in_bytes()),
            d_transform: MatmulDOps::none(),
            gather_indices: None::<&MetalBuffer>,
            m: call_rows as u32,
            n: N as u32,
            k: K as u32,
        };
        let engine = if context.supports_mxu {
            GemmEngine::Mxu
        } else {
            GemmEngine::Simdgroup
        };
        let result = match arm {
            Arm::QuantizedUnsplitGemm => kernel.encode_with_unsplit_gemm_engine(arguments, engine, &mut commands),
            Arm::QuantizedSplitGemm(partitions) => {
                kernel.encode_with_gemm_partition_count(arguments, engine, partitions, &mut commands)
            },
            _ => kernel.encode(arguments, &mut commands),
        };
        result.unwrap_or_else(|_| panic!("projection dispatch failed"));
    }
    let mut readback = context
        .create_buffer(rows * N * dtype.size_in_bytes())
        .unwrap_or_else(|_| panic!("projection readback allocation failed"));
    commands.encode_copy(&output, &mut readback);
    commands.end_encoding().submit().wait_until_completed().unwrap_or_else(|_| panic!("projection execution failed"));
    match dtype {
        DataType::F32 => (&readback).as_slice::<f32>().iter().map(|&x| f64::from(x)).collect(),
        DataType::BF16 => (&readback).as_slice::<bf16>().iter().map(|x| f64::from(x.to_f32())).collect(),
        _ => unreachable!(),
    }
}

#[stark_test]
fn split_k_partial_and_unsplit_output_kernel_caches_remain_distinct_metal() {
    let context = MetalContext::new().unwrap_or_else(|_| panic!("projection context unavailable"));
    let residual = 1.0_f64 / 512.0;
    for bits in [4, 8] {
        let projection = Projection::cancellation(&context, bits);
        for rows in [16, 32] {
            for dtype in [DataType::BF16, DataType::F32] {
                let mut kernel = MatrixKernel::new(&context, DataType::BF16, DataType::BF16, dtype)
                    .unwrap_or_else(|_| panic!("projection kernel construction failed"));
                let arithmetic = (gamma(K) * (2.0 + residual)).next_up();
                let budget = (arithmetic + output_rounding(dtype, (residual + arithmetic).next_up())).next_up();
                // One Matmul/GemmKernel instance owns both the F32 partial
                // pipeline and requested-output pipeline for identical shapes.
                // Exercise both cache-hit orders, then revisit split2.
                for (pass, arm) in [
                    Arm::QuantizedSplitGemm(2),
                    Arm::QuantizedUnsplitGemm,
                    Arm::QuantizedSplitGemm(4),
                    Arm::QuantizedUnsplitGemm,
                    Arm::QuantizedSplitGemm(2),
                ]
                .into_iter()
                .enumerate()
                {
                    let actual = execute_with_kernel(&context, &projection, rows, dtype, arm, &mut kernel);
                    let mut check = Check::default();
                    let mut exact_mismatches = 0_usize;
                    for (index, &value) in actual.iter().enumerate() {
                        let expected = if (index % N).is_multiple_of(2) {
                            residual
                        } else {
                            -residual
                        };
                        check.add(value, expected, budget);
                        exact_mismatches += usize::from(value != expected);
                    }
                    eprintln!(
                        "split_k_cache_anchor bits={bits} rows={rows} dtype={dtype:?} pass={pass} count={} failures={} exact_mismatches={exact_mismatches} max_abs={:.9e} budget_ratio={:.9e}",
                        check.count, check.failures, check.max_error, check.max_budget_ratio
                    );
                    assert!(
                        check.failures == 0 && exact_mismatches == 0,
                        "same-instance split/unsplit kernel cache changed cancellation output"
                    );
                }
            }
        }
    }
}

fn gamma(operations: usize) -> f64 {
    let nu = operations as f64 * 2.0_f64.powi(-24);
    assert!(nu < 1.0);
    (nu / (1.0 - nu)).next_up()
}

fn output_rounding(
    dtype: DataType,
    magnitude_bound: f64,
) -> f64 {
    if dtype == DataType::BF16 {
        // RNE relative-error bound plus half the minimum BF16 subnormal.
        (magnitude_bound * 2.0_f64.powi(-8) + 2.0_f64.powi(-134)).next_up()
    } else {
        // F32 accumulator already includes the final stored F32 rounding.
        0.0
    }
}

#[derive(Default)]
struct Check {
    count: usize,
    failures: usize,
    max_error: f64,
    max_budget_ratio: f64,
}
impl Check {
    fn add(
        &mut self,
        actual: f64,
        expected: f64,
        budget: f64,
    ) {
        self.count += 1;
        let error = (actual - expected).abs();
        self.failures += usize::from(!actual.is_finite() || error > budget);
        self.max_error = self.max_error.max(error);
        self.max_budget_ratio = self.max_budget_ratio.max(error / budget.max(f64::MIN_POSITIVE));
    }
}

#[stark_test]
fn standard_quantized_projection_branch_matched_oracles_metal() {
    let context = MetalContext::new().unwrap_or_else(|_| panic!("projection context unavailable"));
    for bits in [4, 8] {
        for exact in [true, false] {
            let projection = Projection::new(&context, bits, exact);
            for rows in [16, 32] {
                for dtype in [DataType::F32, DataType::BF16] {
                    // Nongathered M>8 cannot select GEMV. The M1 arm is the
                    // standard quantized GEMV; both use the same quant planes.
                    let gemv = execute(&context, &projection, rows, dtype, Arm::QuantizedGemv);
                    let gemm = execute(&context, &projection, rows, dtype, Arm::QuantizedGemm);
                    let unsplit = execute(&context, &projection, rows, dtype, Arm::QuantizedUnsplitGemm);
                    let split2 = execute(&context, &projection, rows, dtype, Arm::QuantizedSplitGemm(2));
                    let split4 = execute(&context, &projection, rows, dtype, Arm::QuantizedSplitGemm(4));
                    let dense = execute(&context, &projection, rows, dtype, Arm::DenseStagedGemm);
                    let (mut gv, mut gm, mut dg, mut cross, mut staged_pair) =
                        (Check::default(), Check::default(), Check::default(), Check::default(), Check::default());
                    let mut max_staging_budget = 0.0_f64;
                    let mut unsplit_check = Check::default();
                    let mut split_checks = [Check::default(), Check::default()];
                    let mut failed_rows = vec![false; rows];
                    let mut failed_columns = [false; N];
                    let mut tile_failures = [0_usize; 5];
                    let mut split_round_matches = [0_usize; 2];
                    for row in 0..rows {
                        for column in 0..N {
                            let (mut affine, mut staged, mut absolute_staged, mut absolute_factors, mut staging) =
                                (0.0, 0.0, 0.0, 0.0, 0.0);
                            for inner in 0..K {
                                let a = f64::from(projection.input[row * K + inner].to_f32());
                                let at = column * K + inner;
                                let w = projection.affine[at];
                                let wb = f64::from(projection.staged[at].to_f32());
                                affine += a * w;
                                staged += a * wb;
                                absolute_staged += (a * wb).abs();
                                absolute_factors += a.abs() * projection.factor_magnitudes[at];
                                staging += a.abs() * (wb - w).abs();
                            }
                            // All fixture products/sums are exact in F64 and
                            // normal in F32. BF16*BF16 products are exact F32.
                            // GEMM's reduction path has at most K roundings.
                            let gemm_arithmetic = (gamma(K) * absolute_staged).next_up();
                            // GEMV: at most K partial/input-sum additions,
                            // at most 2K affine fold FMAs, then five SIMD
                            // reduction additions along any arithmetic path.
                            // The factor magnitude retains code/bias
                            // cancellation instead of hiding it in |weight|.
                            let gemv_arithmetic = (gamma(3 * K + 5) * absolute_factors).next_up();
                            let gm_budget = (gemm_arithmetic
                                + output_rounding(dtype, (staged.abs() + gemm_arithmetic).next_up()))
                            .next_up();
                            let gv_budget = (gemv_arithmetic
                                + output_rounding(dtype, (affine.abs() + gemv_arithmetic).next_up()))
                            .next_up();
                            let at = row * N + column;
                            unsplit_check.add(unsplit[at], staged, gm_budget);
                            split_checks[0].add(split2[at], staged, gm_budget);
                            split_checks[1].add(split4[at], staged, gm_budget);
                            if (gemm[at] - staged).abs() > gm_budget {
                                failed_rows[row] = true;
                                failed_columns[column] = true;
                                tile_failures[column / 16] += 1;
                            }
                            if dtype == DataType::BF16 {
                                for (arm, partitions) in [2, 4].into_iter().enumerate() {
                                    let mut sum = 0.0_f32;
                                    for partition in 0..partitions {
                                        let mut partial = 0.0_f64;
                                        for inner in partition * K / partitions..(partition + 1) * K / partitions {
                                            partial += f64::from(projection.input[row * K + inner].to_f32())
                                                * f64::from(projection.staged[column * K + inner].to_f32());
                                        }
                                        sum += bf16::from_f32(partial as f32).to_f32();
                                    }
                                    split_round_matches[arm] +=
                                        usize::from(f64::from(bf16::from_f32(sum).to_f32()) == gemm[at]);
                                }
                            }
                            gm.add(gemm[at], staged, gm_budget);
                            gv.add(gemv[at], affine, gv_budget);
                            dg.add(dense[at], staged, gm_budget);
                            cross.add(gemm[at], gemv[at], ((staging + gm_budget).next_up() + gv_budget).next_up());
                            staged_pair.add(gemm[at], dense[at], (2.0 * gm_budget).next_up());
                            max_staging_budget = max_staging_budget.max(staging);
                        }
                    }
                    eprintln!(
                        "projection_policy bits={bits} exact={exact} rows={rows} dtype={dtype:?} count={} gemv_max_abs={:.9e} gemm_max_abs={:.9e} dense_max_abs={:.9e} staging_budget_max={max_staging_budget:.9e} gemv_budget_ratio={:.9e} gemm_budget_ratio={:.9e} dense_budget_ratio={:.9e} cross_budget_ratio={:.9e} staged_pair_budget_ratio={:.9e}",
                        gm.count,
                        gv.max_error,
                        gm.max_error,
                        dg.max_error,
                        gv.max_budget_ratio,
                        gm.max_budget_ratio,
                        dg.max_budget_ratio,
                        cross.max_budget_ratio,
                        staged_pair.max_budget_ratio
                    );
                    eprintln!(
                        "projection_split_attribution bits={bits} exact={exact} rows={rows} dtype={dtype:?} unsplit_failures={} unsplit_max_abs={:.9e} unsplit_budget_ratio={:.9e} failed_row_count={} failed_column_count={} column_tiles16_failures={tile_failures:?} split2_round_exact_matches={} split4_round_exact_matches={}",
                        unsplit_check.failures,
                        unsplit_check.max_error,
                        unsplit_check.max_budget_ratio,
                        failed_rows.iter().filter(|&&x| x).count(),
                        failed_columns.iter().filter(|&&x| x).count(),
                        split_round_matches[0],
                        split_round_matches[1]
                    );
                    assert!(unsplit_check.failures == 0, "unsplit GEMM exceeded unchanged branch arithmetic bound");
                    eprintln!(
                        "projection_fp32_partials bits={bits} exact={exact} rows={rows} dtype={dtype:?} split2_failures={} split4_failures={} split2_budget_ratio={:.9e} split4_budget_ratio={:.9e}",
                        split_checks[0].failures,
                        split_checks[1].failures,
                        split_checks[0].max_budget_ratio,
                        split_checks[1].max_budget_ratio
                    );
                    assert!(
                        split_checks.iter().all(|check| check.failures == 0),
                        "forced split-K GEMM exceeded unchanged branch arithmetic bound"
                    );
                    assert!(
                        gv.failures + gm.failures + dg.failures + cross.failures + staged_pair.failures == 0,
                        "projection arithmetic exceeded derived branch bounds: bits={bits} exact={exact} rows={rows} dtype={dtype:?} gemv_failures={} gemm_failures={} dense_failures={} cross_failures={} staged_pair_failures={}",
                        gv.failures,
                        gm.failures,
                        dg.failures,
                        cross.failures,
                        staged_pair.failures
                    );
                }
            }
        }
    }
}

#[stark_test]
fn split_k_cancellation_preserves_f32_partials_and_rejects_bf16_partials_metal() {
    let context = MetalContext::new().unwrap_or_else(|_| panic!("projection context unavailable"));
    let residual = 1.0_f64 / 512.0;
    let legacy = f64::from(bf16::from_f32(1.0 + residual as f32).to_f32() - 1.0);
    for bits in [4, 8] {
        let projection = Projection::cancellation(&context, bits);
        for rows in [16, 32] {
            for dtype in [DataType::F32, DataType::BF16] {
                let arithmetic = (gamma(K) * (2.0 + residual)).next_up();
                let budget = (arithmetic + output_rounding(dtype, (residual + arithmetic).next_up())).next_up();
                assert!((legacy - residual).abs() > budget, "negative control did not expose BF16 partial rounding");
                for partitions in [2, 4] {
                    let actual = execute(&context, &projection, rows, dtype, Arm::QuantizedSplitGemm(partitions));
                    let mut check = Check::default();
                    let mut exact_mismatches = 0_usize;
                    for (index, &value) in actual.iter().enumerate() {
                        let expected = if (index % N).is_multiple_of(2) {
                            residual
                        } else {
                            -residual
                        };
                        check.add(value, expected, budget);
                        exact_mismatches += usize::from(value != expected);
                    }
                    eprintln!(
                        "split_k_cancellation bits={bits} rows={rows} dtype={dtype:?} partitions={partitions} count={} failures={} max_abs={:.9e} budget_ratio={:.9e} legacy_negative_control_rejected=true",
                        check.count, check.failures, check.max_error, check.max_budget_ratio
                    );
                    assert!(
                        check.failures == 0 && exact_mismatches == 0,
                        "split-K discarded an exactly representable cancellation residual"
                    );
                }
            }
        }
    }
}
