// SPDX-License-Identifier: Apache-2.0
//! Original Inkling 2 Mini vision tower and still-image processor.
//! Geometry, layout, and precision follow the pinned Ollama implementation
//! at 6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8.

use std::{collections::HashMap, path::Path, ptr::NonNull, time::Instant};

use bytemuck::{Pod, Zeroable};
use half::bf16;
use metal::{
    MTLBuffer, MTLCompileOptions, MTLComputeCommandEncoder, MTLDevice, MTLDeviceExt, MTLGPUFamily, MTLLanguageVersion,
    MTLLibraryExt, MTLMathMode, MTLResourceOptions, MTLSize,
};
use objc2::runtime::ProtocolObject;

use super::{Binding, Buffer, Device, Pipeline, Result, Runtime, Tensors, WIDTH, execute, zeroed};
use crate::{
    backends::common::InvalidInput,
    engine::vision::preprocess::{PreprocessLimits, RgbImage, decode_inkling_image},
};

pub const INKLING_IMAGE_TOKEN: u32 = 200_054;

const PATCH: usize = 20;
const MERGE: usize = 2;
const TILE: usize = PATCH * MERGE;
const PATCH_WIDTH: usize = PATCH * PATCH * 3;
const VISUAL_WIDTH: usize = 1024;
const HEADS: usize = 16;
const HEAD_DIM: usize = VISUAL_WIDTH / HEADS;
const INTERMEDIATE: usize = 4096;
const POSITION_SIDE: usize = 32;
const LAYERS: usize = 23;
const MAX_PATCHES: usize = 4096;

const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

/// Request-owned rows that remain resident on the decoder's Metal device.
pub struct ImageEmbeddings {
    values: Buffer,
    geometry: Geometry,
}

impl ImageEmbeddings {
    pub fn token_count(&self) -> usize {
        self.geometry.tokens()
    }

    /// Temporal, patch-row and patch-column counts before spatial merging.
    pub fn grid(&self) -> [u32; 3] {
        [1, self.geometry.patch_rows() as u32, self.geometry.patch_columns() as u32]
    }

    /// Encoded image height and width, before padding.
    pub fn source_dimensions(&self) -> [u32; 2] {
        [self.geometry.height, self.geometry.width]
    }

    /// Image height and width after right/bottom padding to the tile grid.
    pub fn processed_dimensions(&self) -> [u32; 2] {
        [(self.geometry.patch_rows() * PATCH) as u32, (self.geometry.patch_columns() * PATCH) as u32]
    }

    pub(super) fn row(
        &self,
        index: usize,
    ) -> Result<Binding<'_>> {
        if index >= self.token_count() {
            return Err("Inkling image embedding row out of bounds");
        }
        Ok((&self.values, index * WIDTH * size_of::<u16>()))
    }
}

#[derive(Clone, Copy)]
struct Geometry {
    width: u32,
    height: u32,
    rows: usize,
    columns: usize,
}

impl Geometry {
    fn new(
        width: u32,
        height: u32,
        available: usize,
    ) -> std::result::Result<Self, InvalidInput> {
        let rows = (height as usize).div_ceil(TILE);
        let columns = (width as usize).div_ceil(TILE);
        let max_tokens = available.min(MAX_PATCHES / (MERGE * MERGE));
        if rows == 0 || columns == 0 || rows > max_tokens / columns {
            return Err(InvalidInput("Inkling image exceeds patch or context budget"));
        }
        Ok(Self {
            width,
            height,
            rows,
            columns,
        })
    }

    fn tokens(&self) -> usize {
        self.rows * self.columns
    }

    fn patch_rows(&self) -> usize {
        self.rows * MERGE
    }

    fn patch_columns(&self) -> usize {
        self.columns * MERGE
    }

    fn patches(&self) -> usize {
        self.patch_rows() * self.patch_columns()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PositionSample {
    indices: [u32; 4],
    coefficients: [f32; 4],
}

fn round(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn pixels(
    image: &RgbImage,
    geometry: Geometry,
) -> Vec<u16> {
    // Quantize CLIP-normalized input first, then use the source's BF16
    // denormalization to [-1, 1]. Padding uses encoded channel value -1.
    let lookup: [[u16; 257]; 3] = std::array::from_fn(|c| {
        std::array::from_fn(|value| {
            let value = if value == 256 {
                -1.0
            } else {
                value as f32
            };
            let normalized = round((value / 255.0 - MEAN[c]) / STD[c]);
            let scaled = round(normalized * round(STD[c]));
            let biased = round(scaled + round(MEAN[c]));
            bf16::from_f32(round(biased * 2.0) - 1.0).to_bits()
        })
    });
    let mut output = Vec::with_capacity(geometry.patches() * PATCH_WIDTH);
    let width = geometry.width as usize;
    let height = geometry.height as usize;
    for py in 0..geometry.patch_rows() {
        for px in 0..geometry.patch_columns() {
            for y in py * PATCH..(py + 1) * PATCH {
                for x in px * PATCH..(px + 1) * PATCH {
                    if y < height && x < width {
                        let offset = (y * width + x) * 3;
                        for (channel, table) in lookup.iter().enumerate() {
                            output.push(table[usize::from(image.pixels()[offset + channel])]);
                        }
                    } else {
                        output.extend(lookup.iter().map(|table| table[256]));
                    }
                }
            }
        }
    }
    output
}

fn positions(geometry: Geometry) -> Vec<PositionSample> {
    let coordinate = |index: usize, extent: usize| {
        let continuous =
            ((index as f64 + 0.5) * POSITION_SIDE as f64 / extent as f64 - 0.5).clamp(0.0, (POSITION_SIDE - 1) as f64);
        let lower = continuous as usize;
        let frac = (continuous - lower as f64) as f32;
        (lower, (lower + 1).min(POSITION_SIDE - 1), frac)
    };
    let mut output = Vec::with_capacity(geometry.patches());
    for py in 0..geometry.patch_rows() {
        let (y0, y1, dy) = coordinate(py, geometry.patch_rows());
        for px in 0..geometry.patch_columns() {
            let (x0, x1, dx) = coordinate(px, geometry.patch_columns());
            output.push(PositionSample {
                indices: [
                    y0 * POSITION_SIDE + x0,
                    y0 * POSITION_SIDE + x1,
                    y1 * POSITION_SIDE + x0,
                    y1 * POSITION_SIDE + x1,
                ]
                .map(|i| i as u32),
                coefficients: [(1.0 - dy) * (1.0 - dx), (1.0 - dy) * dx, dy * (1.0 - dx), dy * dx],
            });
        }
    }
    output
}

fn phases(geometry: Geometry) -> Vec<[f32; 2]> {
    let frequencies: [f32; HEAD_DIM / 4] =
        std::array::from_fn(|i| 10_000f64.powf(-4.0 * i as f64 / HEAD_DIM as f64) as f32);
    let mut output = vec![[1.0, 0.0]; HEAD_DIM / 2];
    output.reserve(geometry.patches() * (HEAD_DIM / 2));
    for row in 0..geometry.patch_rows() {
        for column in 0..geometry.patch_columns() {
            for coordinate in [column + 1, row + 1] {
                for frequency in frequencies {
                    let (sine, cosine) = (coordinate as f32 * frequency).sin_cos();
                    output.push([cosine, sine]);
                }
            }
        }
    }
    output
}

fn upload<T: Pod>(
    device: &Device,
    values: &[T],
) -> Result<Buffer> {
    let bytes = bytemuck::cast_slice::<T, u8>(values);
    let buffer = device
        .new_buffer(bytes.len(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .ok_or("Inkling image buffer allocation failed")?;
    // Uploads are private to the request and complete before encoding.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents().as_ptr().cast(), bytes.len()) };
    Ok(buffer)
}

struct Linear {
    weights: Buffer,
    bias: Option<Buffer>,
    input: usize,
    output: usize,
}

struct Norm {
    weight: Buffer,
    bias: Buffer,
}

struct Block {
    attention_norm: Norm,
    ff_norm: Norm,
    projections: [Linear; 3],
    attention: Linear,
    ff1: Linear,
    ff2: Linear,
    attention_scale: Buffer,
    ff_scale: Buffer,
}

pub(super) struct Vision {
    patch: Linear,
    cls: Buffer,
    cls_position: Buffer,
    positions: Buffer,
    initial_norm: Norm,
    blocks: Vec<Block>,
    projector: Linear,
    output: Linear,
    pipelines: HashMap<&'static str, Pipeline>,
    mpp: bool,
}

impl Vision {
    pub(super) fn load(
        source: &Tensors,
        device: &Device,
    ) -> Result<Self> {
        let weight =
            |name: &str, shape: &[usize]| source.buffer(device, &format!("model.visual.{name}"), "BF16", shape);
        let linear = |name: &str, input, output, bias| -> Result<Linear> {
            Ok(Linear {
                weights: weight(&format!("{name}.weight"), &[output, input])?,
                bias: if bias {
                    Some(weight(&format!("{name}.bias"), &[output])?)
                } else {
                    None
                },
                input,
                output,
            })
        };
        let norm = |name: &str| -> Result<Norm> {
            Ok(Norm {
                weight: weight(&format!("{name}.weight"), &[VISUAL_WIDTH])?,
                bias: weight(&format!("{name}.bias"), &[VISUAL_WIDTH])?,
            })
        };
        let mut blocks = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            let prefix = format!("backbone.layers.{index}");
            let attn = |part| linear(&format!("{prefix}.self_attn.{part}_proj"), VISUAL_WIDTH, VISUAL_WIDTH, true);
            blocks.push(Block {
                attention_norm: norm(&format!("{prefix}.ln_1"))?,
                ff_norm: norm(&format!("{prefix}.ln_2"))?,
                projections: [attn("q")?, attn("k")?, attn("v")?],
                attention: attn("out")?,
                ff1: linear(&format!("{prefix}.mlp.c_fc"), VISUAL_WIDTH, INTERMEDIATE, true)?,
                ff2: linear(&format!("{prefix}.mlp.c_proj"), INTERMEDIATE, VISUAL_WIDTH, true)?,
                attention_scale: weight(&format!("{prefix}.ls_1.gamma"), &[VISUAL_WIDTH])?,
                ff_scale: weight(&format!("{prefix}.ls_2.gamma"), &[VISUAL_WIDTH])?,
            });
        }
        let mut pipelines = HashMap::new();
        let options = MTLCompileOptions::new();
        options.set_math_mode(MTLMathMode::Safe);
        options.set_language_version(MTLLanguageVersion::Version4_0);
        let mut install = |shader, names: &[&'static str]| -> Result<()> {
            let library = device
                .new_library_with_source(shader, Some(&options))
                .map_err(|_| "Inkling vision Metal compilation failed")?;
            for &name in names {
                let function = library.new_function_with_name(name).ok_or("Inkling vision kernel missing")?;
                let pipeline = device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|_| "Inkling vision pipeline creation failed")?;
                pipelines.insert(name, pipeline);
            }
            Ok(())
        };
        install(
            concat!(include_str!("../../backends/metal/kernel/activation/erf.h"), "\n", include_str!("vision.metal")),
            &[
                "inkling_visual_linear",
                "inkling_visual_position",
                "inkling_visual_norm",
                "inkling_visual_rotary",
                "inkling_visual_attention",
                "inkling_visual_residual",
                "inkling_visual_gelu",
                "inkling_visual_merge",
            ],
        )?;
        let mpp = device.supports_family(MTLGPUFamily::Apple10) && device.supports_family(MTLGPUFamily::Metal4);
        if mpp {
            install(include_str!("vision_mpp.metal"), &["inkling_visual_linear_mpp"])?;
        }
        Ok(Self {
            patch: linear("backbone.embeddings.patch_embedding", PATCH_WIDTH, VISUAL_WIDTH, false)?,
            cls: weight("backbone.embeddings.class_embedding", &[VISUAL_WIDTH])?,
            cls_position: weight("backbone.embeddings.cls_position_embedding", &[VISUAL_WIDTH])?,
            positions: weight(
                "backbone.embeddings.position_embedding",
                &[POSITION_SIDE * POSITION_SIDE, VISUAL_WIDTH],
            )?,
            initial_norm: norm("backbone.ln_pre")?,
            blocks,
            projector: linear("proj_in", MERGE * MERGE * VISUAL_WIDTH, INTERMEDIATE, false)?,
            output: linear("proj_out", INTERMEDIATE, WIDTH, false)?,
            pipelines,
            mpp,
        })
    }

    pub(super) fn encode(
        &self,
        runtime: &Runtime,
        bytes: &[u8],
        available_tokens: usize,
    ) -> Result<ImageEmbeddings> {
        let limits = PreprocessLimits {
            max_source_pixels: (MAX_PATCHES * PATCH * PATCH) as u64,
            ..PreprocessLimits::default()
        };
        let image = decode_inkling_image(bytes, &limits, |width, height| {
            Geometry::new(width, height, available_tokens).map(|_| ())
        })
        .map_err(|_| "invalid Inkling image or image patch budget exceeded")?;
        let geometry = Geometry::new(image.width(), image.height(), available_tokens)
            .map_err(|_| "Inkling image patch budget exceeded")?;
        let device = &runtime.device;
        let pixels = upload(device, &pixels(&image, geometry))?;
        let samples = upload(device, &positions(geometry))?;
        let phases = upload(device, &phases(geometry))?;
        let count = geometry.patches() + 1;
        let count_arg = count as u32;
        let bf = |rows, width| zeroed(device, rows * width * size_of::<u16>());
        let hidden = bf(count, VISUAL_WIDTH)?;
        let norm = bf(count, VISUAL_WIDTH)?;
        let projected = bf(count, VISUAL_WIDTH)?;
        let qkv = [bf(count, VISUAL_WIDTH)?, bf(count, VISUAL_WIDTH)?, bf(count, VISUAL_WIDTH)?];
        let attention = bf(count, VISUAL_WIDTH)?;
        let wide = bf(count, INTERMEDIATE)?;
        let merged = bf(geometry.tokens(), MERGE * MERGE * VISUAL_WIDTH)?;
        let values = bf(geometry.tokens(), WIDTH)?;
        execute(runtime, |gpu| {
            let stream = VisionStream {
                encoder: gpu.encoder,
                pipeline: &self.pipelines,
                mpp: self.mpp,
            };
            stream.linear(&self.patch, &pixels, &projected, count - 1);
            stream.dispatch(
                "inkling_visual_position",
                &[&projected, &self.positions, &self.cls, &self.cls_position, &samples, &norm],
                count_arg,
                ((count * VISUAL_WIDTH).div_ceil(128), 1),
                128,
            );
            stream.norm(&self.initial_norm, &norm, &hidden, count);
            for block in &self.blocks {
                stream.norm(&block.attention_norm, &hidden, &norm, count);
                for (matrix, output) in block.projections.iter().zip(&qkv) {
                    stream.linear(matrix, &norm, output, count);
                }
                stream.dispatch(
                    "inkling_visual_rotary",
                    &[&qkv[0], &qkv[1], &phases],
                    count_arg,
                    ((count * VISUAL_WIDTH / 2).div_ceil(128), 1),
                    128,
                );
                stream.dispatch(
                    "inkling_visual_attention",
                    &[&qkv[0], &qkv[1], &qkv[2], &attention],
                    count_arg,
                    (HEADS, count),
                    128,
                );
                stream.linear(&block.attention, &attention, &projected, count);
                stream.residual(&hidden, &projected, &block.attention_scale, count);
                stream.norm(&block.ff_norm, &hidden, &norm, count);
                stream.linear(&block.ff1, &norm, &wide, count);
                stream.gelu(&wide, count * INTERMEDIATE);
                stream.linear(&block.ff2, &wide, &projected, count);
                stream.residual(&hidden, &projected, &block.ff_scale, count);
            }
            stream.dispatch(
                "inkling_visual_merge",
                &[&hidden, &merged],
                [geometry.columns as u32, geometry.rows as u32],
                ((geometry.patches() * VISUAL_WIDTH).div_ceil(128), 1),
                128,
            );
            stream.linear(&self.projector, &merged, &wide, geometry.tokens());
            stream.gelu(&wide, geometry.tokens() * INTERMEDIATE);
            stream.linear(&self.output, &wide, &values, geometry.tokens());
        })?;
        Ok(ImageEmbeddings {
            values,
            geometry,
        })
    }
}

struct VisionStream<'a> {
    encoder: &'a ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &'a HashMap<&'static str, Pipeline>,
    mpp: bool,
}

impl VisionStream<'_> {
    fn dispatch<T>(
        &self,
        name: &str,
        buffers: &[&Buffer],
        params: T,
        groups: (usize, usize),
        threads: usize,
    ) {
        self.encoder.set_compute_pipeline_state(&self.pipeline[name]);
        for (index, buffer) in buffers.iter().enumerate() {
            self.encoder.set_buffer(Some(buffer), 0, index);
        }
        if size_of::<T>() != 0 {
            self.encoder.set_bytes(NonNull::from(&params).cast(), size_of::<T>(), buffers.len());
        }
        self.encoder.dispatch_threadgroups(
            MTLSize {
                width: groups.0,
                height: groups.1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
    }

    fn linear(
        &self,
        linear: &Linear,
        input: &Buffer,
        output: &Buffer,
        rows: usize,
    ) {
        let (name, groups, threads) = if self.mpp {
            ("inkling_visual_linear_mpp", (linear.output.div_ceil(32), rows.div_ceil(16)), 32)
        } else {
            ("inkling_visual_linear", (linear.output.div_ceil(4), rows), 128)
        };
        self.dispatch(
            name,
            &[input, &linear.weights, linear.bias.as_ref().unwrap_or(&linear.weights), output],
            [linear.input as u32, linear.output as u32, rows as u32, u32::from(linear.bias.is_some())],
            groups,
            threads,
        );
    }

    fn norm(
        &self,
        norm: &Norm,
        input: &Buffer,
        output: &Buffer,
        rows: usize,
    ) {
        self.dispatch("inkling_visual_norm", &[input, &norm.weight, &norm.bias, output], (), (rows, 1), 128);
    }

    fn residual(
        &self,
        state: &Buffer,
        update: &Buffer,
        scale: &Buffer,
        rows: usize,
    ) {
        self.dispatch(
            "inkling_visual_residual",
            &[state, update, scale],
            rows as u32,
            ((rows * VISUAL_WIDTH).div_ceil(128), 1),
            128,
        );
    }

    fn gelu(
        &self,
        values: &Buffer,
        elements: usize,
    ) {
        self.dispatch("inkling_visual_gelu", &[values], elements as u32, (elements.div_ceil(128), 1), 128);
    }
}

/// Check the original tower against an ignored SafeTensors oracle, returning
/// aggregate errors only. Checkpoint and image contents stay process-private.
#[doc(hidden)]
pub fn measure_vision(
    checkpoint: &Path,
    image: &[u8],
    reference: &Path,
) -> Result<serde_json::Value> {
    let runtime = Runtime::new()?;
    let source = Tensors::open(checkpoint)?;
    let tower = Vision::load(&source, &runtime.device)?;
    let start = Instant::now();
    let actual = tower.encode(&runtime, image, MAX_PATCHES / (MERGE * MERGE))?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
    let rows = actual.token_count();
    let oracle = Tensors::open(reference)?;
    let expected = oracle.values::<u16>("features", "BF16", &[rows, WIDTH])?;
    // Encoding has completed the only command before either slice is read.
    let produced = unsafe { std::slice::from_raw_parts(actual.values.contents().as_ptr().cast::<u16>(), rows * WIDTH) };
    let mut different = 0;
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0;
    let mut target_sq = 0.0;
    let mut dot = 0.0;
    let mut actual_sq = 0.0;
    for (&a, &b) in produced.iter().zip(&expected) {
        different += usize::from(a != b);
        let a = f64::from(bf16::from_bits(a).to_f32());
        let b = f64::from(bf16::from_bits(b).to_f32());
        if !a.is_finite() || !b.is_finite() {
            return Err("non-finite Inkling vision embedding");
        }
        let delta = a - b;
        max_abs = max_abs.max(delta.abs());
        sum_sq += delta * delta;
        target_sq += b * b;
        dot += a * b;
        actual_sq += a * a;
    }
    if target_sq == 0.0 || actual_sq == 0.0 {
        return Err("zero Inkling vision embedding");
    }
    Ok(serde_json::json!({
        "image_tokens": rows,
        "grid": actual.grid(),
        "source_dimensions": actual.source_dimensions(),
        "processed_dimensions": actual.processed_dimensions(),
        "different_bf16_words": different,
        "total_bf16_words": produced.len(),
        "max_abs": max_abs,
        "relative_l2": (sum_sq / target_sq).sqrt(),
        "cosine": dot / (target_sq * actual_sq).sqrt(),
        "encode_ms": encode_ms,
        "matrix_pipeline": tower.mpp,
    }))
}
