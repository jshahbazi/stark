//! Native packed-weight GEMV. Only the selected expert matrix is submitted;
//! weights are never expanded into a full-precision model in memory.

use std::sync::OnceLock;

use metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLDeviceExt, MTLLibraryExt, MTLResourceOptions, MTLSize,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{Error, Result, weights};

struct Executor {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    packed: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    affine: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static EXECUTOR: OnceLock<std::result::Result<Executor, &'static str>> = OnceLock::new();

pub(super) fn initialize() -> Result<()> {
    executor().map(|_| ())
}

fn executor() -> Result<&'static Executor> {
    EXECUTOR
        .get_or_init(|| {
            let device = <dyn MTLDevice>::system_default().ok_or("Metal device unavailable")?;
            let queue = device.new_command_queue().ok_or("Metal queue unavailable")?;
            let mut source = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
            for (name, table) in
                [("iq2xxs_grid", weights::IQ2XXS_GRID.as_slice()), ("iq2xs_grid", weights::IQ2XS_GRID.as_slice())]
            {
                source.push_str(&format!("constant ulong {name}[{}]={{", table.len()));
                for value in table {
                    source.push_str(&format!("0x{value:016x}ul,"));
                }
                source.push_str("};\n");
            }
            source.push_str("constant uchar ksigns[128]={");
            for value in weights::KSIGNS_IQ2XS {
                source.push_str(&format!("{value},"));
            }
            source.push_str("};\nconstant short iq4values[16]={");
            for value in weights::KVALUES_IQ4NL {
                source.push_str(&format!("{value},"));
            }
            source.push_str("};\n");
            source.push_str("constant uint iq3xxs_grid[256]={");
            for value in weights::IQ3XXS_GRID {
                source.push_str(&format!("0x{value:08x}u,"));
            }
            source.push_str("};\n");
            source.push_str(include_str!("packed_matvec.metal"));
            let options = metal::MTLCompileOptions::new();
            options.set_math_mode(metal::MTLMathMode::Safe);
            let library = device
                .new_library_with_source(&source, Some(&options))
                .map_err(|_| "Metal kernel compilation failed")?;
            let packed =
                library.new_function_with_name("hybrid_packed_matvec").ok_or("Metal packed function unavailable")?;
            let affine =
                library.new_function_with_name("hybrid_affine_matvec").ok_or("Metal affine function unavailable")?;
            let packed =
                device.new_compute_pipeline_state_with_function(&packed).map_err(|_| "Metal packed pipeline failed")?;
            let affine =
                device.new_compute_pipeline_state_with_function(&affine).map_err(|_| "Metal affine pipeline failed")?;
            // The reduction uses eight 32-lane SIMD groups in a 256-thread group.
            // Reject incompatible execution widths rather than reading unwritten
            // threadgroup partials or submitting an unsupported group size.
            for pipeline in [&packed, &affine] {
                if pipeline.thread_execution_width() != 32 || pipeline.max_total_threads_per_threadgroup() < 256 {
                    return Err("Metal matrix execution width is unsupported");
                }
            }
            Ok(Executor {
                device,
                queue,
                packed,
                affine,
            })
        })
        .as_ref()
        .map_err(|message| Error::Unsupported(message))
}

fn buffer(
    executor: &Executor,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    executor
        .device
        .new_buffer_with_data(bytes, MTLResourceOptions::STORAGE_MODE_SHARED)
        .ok_or(Error::Invalid("Metal matrix buffer allocation failed"))
}

impl Executor {
    fn run(
        &self,
        codes: &[u8],
        input: &[f32],
        parameters: [u32; 8],
        affine: Option<(&[u8], &[u8])>,
    ) -> Result<Vec<f32>> {
        objc2::rc::autoreleasepool(|_| self.run_inner(codes, input, parameters, affine))
    }

    fn run_inner(
        &self,
        codes: &[u8],
        input: &[f32],
        parameters: [u32; 8],
        affine: Option<(&[u8], &[u8])>,
    ) -> Result<Vec<f32>> {
        // These bounded uploads retain compressed storage and are released only
        // after GPU completion. No unowned mmap page or rounded Rust allocation
        // is exposed to Metal through an unsafe no-copy buffer.
        let codes = buffer(self, codes)?;
        let input = buffer(self, bytemuck::cast_slice(input))?;
        let params = buffer(self, bytemuck::cast_slice(&parameters))?;
        let output = self
            .device
            .new_buffer(parameters[0] as usize * 4, MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or(Error::Invalid("Metal readout allocation failed"))?;
        let planes = affine
            .map(|(scales, biases)| Ok::<_, Error>((buffer(self, scales)?, buffer(self, biases)?)))
            .transpose()?;
        let command = self.queue.command_buffer().ok_or(Error::Invalid("Metal command allocation failed"))?;
        let encoding = command.compute_command_encoder().ok_or(Error::Invalid("Metal encoder allocation failed"))?;
        encoding.set_compute_pipeline_state(if planes.is_some() {
            &self.affine
        } else {
            &self.packed
        });
        for (index, value) in [&codes, &input, &output, &params].into_iter().enumerate() {
            encoding.set_buffer(Some(value), 0, index);
        }
        if let Some((scales, biases)) = &planes {
            encoding.set_buffer(Some(scales), 0, 4);
            encoding.set_buffer(Some(biases), 0, 5);
        }
        encoding.dispatch_threadgroups(
            MTLSize {
                width: parameters[0] as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoding.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Invalid("Metal matrix execution failed"));
        }
        // SAFETY: the shared buffer has exactly rows f32 slots. The command has
        // completed and wrote each slot once before this immutable host read.
        let values =
            unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), parameters[0] as usize) }
                .to_vec();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("nonfinite Metal matrix output"));
        }
        Ok(values)
    }
}

fn dimensions(
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Result<()> {
    if rows == 0
        || columns == 0
        || rows > 8_000_000
        || columns > 8_000_000
        || input.len() != columns
        || input.iter().any(|v| !v.is_finite())
    {
        return Err(Error::Invalid("invalid Metal matrix dimensions"));
    }
    Ok(())
}

pub(super) fn matvec(
    bytes: &[u8],
    kind: u32,
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    dimensions(rows, columns, input)?;
    packed_matvec(bytes, kind, rows, columns, input, 0)
}

fn batch_dimensions(
    batches: usize,
    rows: usize,
    columns: usize,
    inputs: &[f32],
) -> Result<usize> {
    let total_rows = batches.checked_mul(rows).ok_or(Error::Invalid("batched Metal rows overflow"))?;
    if batches == 0
        || rows == 0
        || columns == 0
        || total_rows > 8_000_000
        || inputs.len() > 8_000_000
        || batches.checked_mul(columns) != Some(inputs.len())
        || inputs.iter().any(|value| !value.is_finite())
    {
        return Err(Error::Invalid("invalid batched Metal matrix dimensions"));
    }
    Ok(total_rows)
}

pub(super) fn batched_matvec(
    bytes: &[u8],
    kind: u32,
    batches: usize,
    rows: usize,
    columns: usize,
    inputs: &[f32],
) -> Result<Vec<f32>> {
    let total_rows = batch_dimensions(batches, rows, columns, inputs)?;
    packed_matvec(bytes, kind, total_rows, columns, inputs, rows)
}

fn packed_matvec(
    bytes: &[u8],
    kind: u32,
    rows: usize,
    columns: usize,
    input: &[f32],
    rows_per_batch: usize,
) -> Result<Vec<f32>> {
    let (block, block_bytes) = weights::block_layout(kind)?;
    if !matches!(kind, 0 | 1 | 30 | 8 | 10 | 11 | 12 | 13 | 14 | 16 | 17 | 18 | 23)
        || !columns.is_multiple_of(block)
        || rows.checked_mul(columns / block).and_then(|n| n.checked_mul(block_bytes)) != Some(bytes.len())
    {
        return Err(Error::Invalid("invalid packed Metal matrix size"));
    }
    executor()?.run(
        bytes,
        input,
        [rows as u32, columns as u32, kind, block as u32, block_bytes as u32, 0, rows_per_batch as u32, 0],
        None,
    )
}

pub(super) fn mlx_matvec(
    codes: &[u8],
    scales: &[u8],
    biases: &[u8],
    scale_dtype: u32,
    bits: usize,
    group: usize,
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    dimensions(rows, columns, input)?;
    affine_matvec(codes, scales, biases, scale_dtype, bits, group, rows, columns, input, 0)
}

pub(super) fn mlx_batched_matvec(
    codes: &[u8],
    scales: &[u8],
    biases: &[u8],
    scale_dtype: u32,
    bits: usize,
    group: usize,
    batches: usize,
    rows: usize,
    columns: usize,
    inputs: &[f32],
) -> Result<Vec<f32>> {
    let total_rows = batch_dimensions(batches, rows, columns, inputs)?;
    affine_matvec(codes, scales, biases, scale_dtype, bits, group, total_rows, columns, inputs, rows)
}

fn affine_matvec(
    codes: &[u8],
    scales: &[u8],
    biases: &[u8],
    scale_dtype: u32,
    bits: usize,
    group: usize,
    rows: usize,
    columns: usize,
    input: &[f32],
    rows_per_batch: usize,
) -> Result<Vec<f32>> {
    let item = match scale_dtype {
        0 => 4,
        1 | 30 => 2,
        _ => return Err(Error::Invalid("invalid affine scale dtype")),
    };
    if !matches!(bits, 4 | 8) || !matches!(group, 32 | 64 | 128) {
        return Err(Error::Invalid("invalid affine Metal quantization"));
    }
    let plane_size = rows.checked_mul(columns / group).and_then(|n| n.checked_mul(item));
    if !columns.is_multiple_of(group)
        || rows.checked_mul(columns).and_then(|n| n.checked_mul(bits)).map(|n| n / 8) != Some(codes.len())
        || plane_size != Some(scales.len())
        || scales.len() != biases.len()
    {
        return Err(Error::Invalid("invalid affine Metal matrix size"));
    }
    executor()?.run(
        codes,
        input,
        [rows as u32, columns as u32, 0, rows_per_batch as u32, 0, bits as u32, group as u32, scale_dtype],
        Some((scales, biases)),
    )
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn completed_matrix_submissions_release_their_uploaded_weights() {
        let rows = 1024;
        let columns = 1024;
        let bytes = vec![0u8; rows * columns / 32 * 34];
        let input = vec![1.0; columns];
        matvec(&bytes, 8, rows, columns, &input).unwrap();
        let device = &executor().unwrap().device;
        let before = device.current_allocated_size();
        for _ in 0..32 {
            assert_eq!(matvec(&bytes, 8, rows, columns, &input).unwrap().len(), rows);
        }
        let after = device.current_allocated_size();
        // Allow driver scratch/reuse, but reject retaining a new compressed
        // matrix for every completed command in an undrained autorelease pool.
        assert!(after <= before + 16 * 1024 * 1024);
    }

    #[stark_test]
    fn metal_matches_all_checkpoint_quantization_formats() {
        for kind in [0, 1, 30, 8, 10, 11, 12, 13, 14, 16, 17, 18, 23] {
            let (block, size) = weights::block_layout(kind).unwrap();
            let columns = block.max(256);
            let mut bytes = Vec::new();
            for row in 0..3 {
                for number in 0..columns / block {
                    let mut source: Vec<u8> =
                        (0..size).map(|i| ((i * 37 + number * 13 + row * 23) % 256) as u8).collect();
                    let half_offsets: &[usize] = match kind {
                        1 | 8 | 16 | 17 | 18 | 23 => &[0],
                        10 => &[80, 82],
                        11 => &[108],
                        12 | 13 => &[0, 2],
                        14 => &[208],
                        _ => &[],
                    };
                    for &at in half_offsets {
                        source[at..at + 2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
                    }
                    if kind == 0 {
                        source.copy_from_slice(&(0.01 * (number + row + 1) as f32).to_le_bytes());
                    }
                    if kind == 30 {
                        source.copy_from_slice(&half::bf16::from_f32(0.125).to_bits().to_le_bytes());
                    }
                    bytes.extend(source);
                }
            }
            let input: Vec<f32> = (0..columns).map(|i| (i as f32 * 0.17).sin()).collect();
            let actual = matvec(&bytes, kind, 3, columns, &input).unwrap();
            let factors = [1.0f32, -0.5, 2.0];
            let batch_inputs: Vec<f32> =
                factors.iter().flat_map(|factor| input.iter().map(move |x| factor * x)).collect();
            let batched = batched_matvec(&bytes, kind, 3, 1, columns, &batch_inputs).unwrap();
            let row_size = bytes.len() / 3;
            for row in 0..3 {
                let mut expected = 0.0f64;
                let mut absolute = 0.0f64;
                for (source, x) in
                    bytes[row * row_size..(row + 1) * row_size].chunks_exact(size).zip(input.chunks_exact(block))
                {
                    let mut values = vec![0.0; block];
                    weights::decode_block(kind, source, &mut values).unwrap();
                    for (value, input) in values.iter().zip(x) {
                        let p = *value as f64 * *input as f64;
                        expected += p;
                        absolute += p.abs();
                    }
                }
                assert!((actual[row] as f64 - expected).abs() <= 2e-5 * absolute.max(1.0), "format {kind}");
                assert!(
                    (f64::from(batched[row]) - expected * f64::from(factors[row])).abs()
                        <= 2e-5 * (absolute * f64::from(factors[row]).abs()).max(1.0),
                    "batched format {kind}"
                );
            }
        }
    }

    fn float_bytes(
        kind: u32,
        value: f32,
    ) -> Vec<u8> {
        match kind {
            0 => value.to_le_bytes().to_vec(),
            1 => half::f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
            30 => half::bf16::from_f32(value).to_bits().to_le_bytes().to_vec(),
            _ => unreachable!(),
        }
    }

    fn float_roundtrip(
        kind: u32,
        value: f32,
    ) -> f32 {
        match kind {
            0 => value,
            1 => half::f16::from_f32(value).to_f32(),
            30 => half::bf16::from_f32(value).to_f32(),
            _ => unreachable!(),
        }
    }

    #[stark_test]
    fn metal_affine_four_and_eight_bit_planes_match_scalar() {
        for bits in [4, 8] {
            for group in [32, 64, 128] {
                for dtype in [0, 1, 30] {
                    let rows = 3;
                    let columns = 3 * group;
                    let mut codes = vec![0u8; rows * columns * bits / 8];
                    let mut scales = Vec::new();
                    let mut biases = Vec::new();
                    let mut decoded = vec![0.0f32; rows * columns];
                    for row in 0..rows {
                        for block in 0..3 {
                            let scale = 0.123 + 0.02 * (row * 3 + block) as f32;
                            let bias = -0.07 * (row * 3 + block + 1) as f32;
                            scales.extend(float_bytes(dtype, scale));
                            biases.extend(float_bytes(dtype, bias));
                            for column in block * group..(block + 1) * group {
                                let q = ((row * 31 + column * 7) % (1usize << bits)) as u8;
                                let element = row * columns + column;
                                codes[element * bits / 8] |= q << ((column * bits) % 8);
                                decoded[element] =
                                    float_roundtrip(dtype, scale) * f32::from(q) + float_roundtrip(dtype, bias);
                            }
                        }
                    }
                    let input: Vec<f32> = (0..columns).map(|i| (i as f32 * 0.13).cos()).collect();
                    let actual =
                        mlx_matvec(&codes, &scales, &biases, dtype, bits, group, rows, columns, &input).unwrap();
                    let factors = [1.0f32, -0.5, 2.0];
                    let batch_inputs: Vec<f32> =
                        factors.iter().flat_map(|factor| input.iter().map(move |x| factor * x)).collect();
                    let batched =
                        mlx_batched_matvec(&codes, &scales, &biases, dtype, bits, group, 3, 1, columns, &batch_inputs)
                            .unwrap();
                    for row in 0..rows {
                        let products: Vec<f64> = decoded[row * columns..(row + 1) * columns]
                            .iter()
                            .zip(&input)
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .collect();
                        let expected: f64 = products.iter().sum();
                        let absolute: f64 = products.iter().map(|x| x.abs()).sum();
                        assert!(
                            (f64::from(actual[row]) - expected).abs() <= 2e-5 * absolute.max(1.0),
                            "affine bits={bits} group={group} dtype={dtype}"
                        );
                        assert!(
                            (f64::from(batched[row]) - expected * f64::from(factors[row])).abs()
                                <= 2e-5 * (absolute * f64::from(factors[row]).abs()).max(1.0),
                            "batched affine bits={bits} group={group} dtype={dtype}"
                        );
                    }
                }
            }
        }
    }

    #[stark_test]
    fn invalid_affine_geometry_is_rejected_before_gpu_submission() {
        assert!(mlx_matvec(&[0; 16], &[0; 4], &[0; 4], 0, 4, 0, 1, 32, &[0.0; 32]).is_err());
        assert!(mlx_matvec(&[0; 16], &[0; 4], &[0; 4], 0, 0, 32, 1, 32, &[0.0; 32]).is_err());
        assert!(mlx_matvec(&[0; 16], &[0; 4], &[0; 4], 0, 4, 64, 1, 32, &[0.0; 32]).is_err());
        assert!(mlx_matvec(&[0; 16], &[0; 4], &[0; 4], 0, 4, 32, 1, 33, &[0.0; 33]).is_err());
        assert!(mlx_matvec(&[0; 16], &[0; 4], &[0; 4], 0, 4, 32, 1, 32, &[f32::NAN; 32]).is_err());
        assert!(batched_matvec(&[0; 68], 8, 0, 1, 32, &[]).is_err());
        assert!(batched_matvec(&[0; 68], 8, 2, 1, 32, &[0.0; 32]).is_err());
        assert!(batched_matvec(&[0; 68], 8, usize::MAX, 2, 32, &[]).is_err());
        assert!(mlx_batched_matvec(&[0; 32], &[0; 8], &[0; 8], 0, 4, 0, 2, 1, 32, &[0.0; 64]).is_err());
        for kind in [101, 102, 103] {
            assert!(matvec(&[0; 8], kind, 1, 1, &[0.0]).is_err());
            assert!(batched_matvec(&[0; 8], kind, 1, 1, 1, &[0.0]).is_err());
        }
    }

    #[stark_test]
    fn parsed_affine_expert_slices_preserve_independent_planes() {
        let directory = std::env::temp_dir().join(format!(
            "stark-metal-expert-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let mut payload = Vec::new();
        let codes = vec![0x32u8; 2 * 3 * 16];
        payload.extend(&codes);
        let scale_start = payload.len();
        let scales: Vec<f32> = (0..6).map(|i| 0.125 * (i + 1) as f32).collect();
        for scale in &scales {
            payload.extend(scale.to_le_bytes());
        }
        let bias_start = payload.len();
        let biases: Vec<f32> = (0..6).map(|i| -0.25 * i as f32).collect();
        for bias in &biases {
            payload.extend(bias.to_le_bytes());
        }
        let mut header = serde_json::to_vec(&serde_json::json!({
            "experts.weight":{"dtype":"U32","shape":[2,3,4],"data_offsets":[0,scale_start]},
            "experts.scales":{"dtype":"F32","shape":[2,3,1],"data_offsets":[scale_start,bias_start]},
            "experts.biases":{"dtype":"F32","shape":[2,3,1],"data_offsets":[bias_start,payload.len()]},
        }))
        .unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        std::fs::write(directory.join("model.safetensors"), file).unwrap();
        std::fs::write(directory.join("config.json"), b"{\"quantization\":{\"bits\":4,\"group_size\":32}}").unwrap();
        let weights = weights::Weights::open(&directory).unwrap();
        assert_eq!(weights.shape("experts.weight").unwrap(), &[2, 3, 32]);
        let input: Vec<f32> = (0..32).map(|i| 0.125 * (i as f32 - 16.0)).collect();
        for expert in 0..2 {
            let actual = weights.expert_matvec("experts.weight", expert, &input).unwrap();
            for row in 0..3 {
                let index = expert * 3 + row;
                let expected: f32 = input
                    .iter()
                    .enumerate()
                    .map(|(col, &x)| {
                        (scales[index]
                            * if col % 2 == 0 {
                                2.0
                            } else {
                                3.0
                            }
                            + biases[index])
                            * x
                    })
                    .sum();
                assert!((actual[row] - expected).abs() < 1e-5);
            }
        }
        let batch_inputs: Vec<f32> = input.iter().copied().chain(input.iter().map(|x| -0.5 * x)).collect();
        let batched = weights.batched_matvec("experts.weight", &batch_inputs).unwrap();
        assert_eq!(batched.len(), 6);
        for expert in 0..2 {
            let single =
                weights.expert_matvec("experts.weight", expert, &batch_inputs[expert * 32..(expert + 1) * 32]).unwrap();
            for row in 0..3 {
                assert_eq!(batched[expert * 3 + row], single[row]);
            }
        }
        drop(weights);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[stark_test]
    fn native_head_batching_matches_separate_dispatches() {
        let (batches, rows, columns) = (64, 512, 256);
        let row_bytes = columns / 32 * 34;
        let matrix_bytes = rows * row_bytes;
        let mut bytes = vec![0u8; batches * matrix_bytes];
        for (index, block) in bytes.chunks_exact_mut(34).enumerate() {
            block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
            for (i, code) in block[2..].iter_mut().enumerate() {
                *code = ((index * 7 + i * 3) % 255) as u8;
            }
        }
        let inputs: Vec<f32> = (0..batches * columns).map(|i| (i as f32 * 0.011).sin()).collect();
        // Warm compilation and allocation before measuring the two execution
        // shapes. Only aggregate timings are reported; fixtures are synthetic.
        let batched = batched_matvec(&bytes, 8, batches, rows, columns, &inputs).unwrap();
        let start = std::time::Instant::now();
        let mut separate = Vec::with_capacity(batches * rows);
        for batch in 0..batches {
            separate.extend(
                matvec(
                    &bytes[batch * matrix_bytes..(batch + 1) * matrix_bytes],
                    8,
                    rows,
                    columns,
                    &inputs[batch * columns..(batch + 1) * columns],
                )
                .unwrap(),
            );
        }
        let separate_elapsed = start.elapsed();
        assert_eq!(batched, separate);
        let start = std::time::Instant::now();
        for _ in 0..3 {
            let result = batched_matvec(&bytes, 8, batches, rows, columns, &inputs).unwrap();
            assert_eq!(result, separate);
        }
        let batch_elapsed = start.elapsed() / 3;
        eprintln!(
            "native_hybrid_head_batching heads={batches} rows={rows} columns={columns} separate_ms={:.3} batched_ms={:.3}",
            separate_elapsed.as_secs_f64() * 1000.0,
            batch_elapsed.as_secs_f64() * 1000.0
        );
    }
}
