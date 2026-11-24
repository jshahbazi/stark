// SPDX-License-Identifier: Apache-2.0
//! Affine 8-bit dense weights of the Inkling 2 Mini affine8 artifact.
//!
//! The artifact stores each quantized original tensor `NAME` as
//! `NAME.affine8_codes` (U8 [rows, columns]), `NAME.affine8_scales` and
//! `NAME.affine8_biases` (BF16 [rows, columns / 64]). The effective weight is
//! `bf16(fma(code, scale, bias))` evaluated in FP32. Every consumer of a dense
//! tensor computes with exactly these values: prefill reads a BF16 copy made at
//! load, and the decode kernels in `affine8.metal` dequantize the codes to the
//! same BF16 values inside the unchanged per-row reduction order, so each
//! decode output equals the BF16 kernel's output on the copy.

use std::collections::HashMap;

use half::bf16;

use super::*;

pub(super) const GROUP: usize = 64;
const CODES: &str = "affine8_codes";
const SCALES: &str = "affine8_scales";
const BIASES: &str = "affine8_biases";
/// The probe tensor whose presence marks an affine8 checkpoint.
const PROBE: &str = "model.llm.layers.0.attn.wq_du.weight";

pub(super) const KERNELS: [Kernel; 11] = [
    "ink_a8_matvec1",
    "ink_a8_matvec2",
    "ink_a8_matvec3",
    "ink_a8_matvec4",
    "ink_a8_matvec_visit",
    "ink_a8_unembed1",
    "ink_a8_unembed2",
    "ink_a8_unembed3",
    "ink_a8_unembed4",
    "ink_a8_dequantize",
    "ink_a8_embedding",
];
pub(super) const DRAFT_EMBEDDING: Kernel = "ink_a8_embedding_draft_ids";

/// Whether the checkpoint carries affine 8-bit dense tensors instead of BF16.
pub(super) fn present(source: &Tensors) -> bool {
    source.contains(&format!("{PROBE}.{CODES}")) && !source.contains(PROBE)
}

pub(super) fn register(runtime: &mut Runtime) -> Result<()> {
    if runtime.pipelines.contains_key(DRAFT_EMBEDDING) {
        return Ok(());
    }
    let options = MTLCompileOptions::new();
    options.set_math_mode(MTLMathMode::Safe);
    options.set_language_version(MTLLanguageVersion::Version4_0);
    let sources = [
        (include_str!("affine8.metal").to_owned(), &KERNELS[..]),
        (
            concat!(
                "#define INK_A8_DRAFT_EMBEDDING\n#define ink_a8_embedding ink_a8_embedding_draft_ids\n",
                include_str!("affine8.metal")
            )
            .to_owned(),
            &[DRAFT_EMBEDDING][..],
        ),
    ];
    for (source, names) in sources {
        let library = runtime
            .device
            .new_library_with_source(&source, Some(&options))
            .map_err(|_| "Inkling affine8 Metal library compilation failed")?;
        for &name in names {
            let function = library.new_function_with_name(name).ok_or(name)?;
            let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
            runtime.pipelines.insert(name, pipeline);
        }
    }
    Ok(())
}

/// Compile `source` and register `names` (diagnostics and decode kernels).
pub(super) fn compile(
    runtime: &mut Runtime,
    source: &str,
    names: &[Kernel],
) -> Result<()> {
    let options = MTLCompileOptions::new();
    options.set_math_mode(MTLMathMode::Safe);
    options.set_language_version(MTLLanguageVersion::Version4_0);
    let library = runtime
        .device
        .new_library_with_source(source, Some(&options))
        .map_err(|_| "Inkling Metal compilation failed")?;
    for &name in names {
        let function = library.new_function_with_name(name).ok_or(name)?;
        let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
        runtime.pipelines.insert(name, pipeline);
    }
    Ok(())
}

/// K lanes per output row, as `ink_matvec_source` chooses from the output count.
pub(super) fn lanes(rows: usize) -> usize {
    if rows <= 64 {
        32
    } else {
        16
    }
}

/// Interleaved vec4 position of input vec4 column `c` (see `affine8.metal`).
pub(super) fn position(
    c: usize,
    cols: usize,
    rows: usize,
) -> usize {
    let lanes = lanes(rows);
    let cols4 = cols / 4;
    let stride = lanes * 8;
    let end = cols4 - cols4 % stride;
    if c < end {
        let r = c % stride;
        return c - r + (r % lanes) * 8 + r / lanes;
    }
    let t = c - end;
    let tail = (cols4 - end) / lanes;
    end + (t % lanes) * tail + t / lanes
}

/// The effective weight `bf16(fma(code, scale, bias))` from raw planes.
pub(super) fn effective(
    code: u8,
    scale: u16,
    bias: u16,
) -> bf16 {
    bf16::from_f32(f32::from(code).mul_add(bf16::from_bits(scale).to_f32(), bf16::from_bits(bias).to_f32()))
}

/// The unrounded FP32 dequantized value, for references only.
pub(super) fn exact(
    code: u8,
    scale: u16,
    bias: u16,
) -> f32 {
    f32::from(code).mul_add(bf16::from_bits(scale).to_f32(), bf16::from_bits(bias).to_f32())
}

/// Raw affine planes of one tensor, as stored in the artifact.
pub(super) struct Planes {
    pub(super) codes: Vec<u8>,
    pub(super) scales: Vec<u16>,
    pub(super) biases: Vec<u16>,
    pub(super) rows: usize,
    pub(super) cols: usize,
}

impl Planes {
    pub(super) fn read(
        source: &Tensors,
        name: &str,
        shape: [usize; 2],
    ) -> Result<Self> {
        let [rows, cols] = shape;
        if cols % GROUP != 0 || cols % 4 != 0 {
            return Err("Inkling affine8 tensor width is not a whole number of groups");
        }
        let groups = cols / GROUP;
        Ok(Self {
            codes: source.values(&format!("{name}.{CODES}"), "U8", &[rows, cols])?,
            scales: source.values(&format!("{name}.{SCALES}"), "BF16", &[rows, groups])?,
            biases: source.values(&format!("{name}.{BIASES}"), "BF16", &[rows, groups])?,
            rows,
            cols,
        })
    }

    fn param(
        &self,
        row: usize,
        col: usize,
    ) -> (u16, u16) {
        let index = row * (self.cols / GROUP) + col / GROUP;
        (self.scales[index], self.biases[index])
    }

    /// BF16 effective weights of the leading `rows` rows, padded with zeros
    /// to `padded` rows, written into a new shared buffer.
    pub(super) fn effective_buffer(
        &self,
        device: &Device,
        rows: usize,
        padded: usize,
    ) -> Result<Buffer> {
        if rows > self.rows || padded < rows {
            return Err("invalid Inkling affine8 row range");
        }
        let buffer = zeroed(device, padded * self.cols * 2)?;
        // Exclusively initialized here, before any Metal dispatch reads it.
        let out = unsafe { std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<u16>(), rows * self.cols) };
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).min(16);
        let chunk = rows.div_ceil(threads).max(1) * self.cols;
        std::thread::scope(|scope| {
            for (index, part) in out.chunks_mut(chunk).enumerate() {
                scope.spawn(move || {
                    let first = index * chunk;
                    for (offset, value) in part.iter_mut().enumerate() {
                        let element = first + offset;
                        let (row, col) = (element / self.cols, element % self.cols);
                        let (scale, bias) = self.param(row, col);
                        *value = effective(self.codes[element], scale, bias).to_bits();
                    }
                });
            }
        });
        Ok(buffer)
    }

    /// FP32 values of one row: BF16 effective weights, or the unrounded
    /// dequantized values when `rounded` is false.
    pub(super) fn row(
        &self,
        row: usize,
        rounded: bool,
    ) -> Vec<f32> {
        (0..self.cols)
            .map(|col| {
                let (scale, bias) = self.param(row, col);
                let code = self.codes[row * self.cols + col];
                if rounded {
                    effective(code, scale, bias).to_f32()
                } else {
                    exact(code, scale, bias)
                }
            })
            .collect()
    }
}

/// One affine tensor in the kernel layout: interleaved codes, then one
/// `u32` per row and group holding the BF16 scale (low) and bias (high).
pub(super) struct Affine8 {
    pub(super) tensor: Buffer,
    pub(super) rows: usize,
    pub(super) cols: usize,
}

impl Affine8 {
    /// Lay out the leading `rows` rows of `planes` for the decode kernels.
    pub(super) fn new(
        device: &Device,
        planes: &Planes,
        rows: usize,
    ) -> Result<Self> {
        let cols = planes.cols;
        if rows > planes.rows {
            return Err("invalid Inkling affine8 layout");
        }
        let cols4 = cols / 4;
        let end = cols4 - cols4 % (lanes(rows) * 8);
        if !(cols4 - end).is_multiple_of(lanes(rows)) {
            return Err("Inkling affine8 tail is not a whole number of lanes");
        }
        let groups = cols / GROUP;
        let bytes = rows * cols + rows * groups * 4;
        let tensor = zeroed(device, bytes)?;
        // Exclusively initialized here, before any Metal dispatch reads it.
        let out = unsafe { std::slice::from_raw_parts_mut(tensor.contents().as_ptr().cast::<u8>(), bytes) };
        let (codes, params) = out.split_at_mut(rows * cols);
        let order = (0..cols4).map(|c| position(c, cols, rows)).collect::<Vec<_>>();
        for (row, dest) in codes.chunks_exact_mut(cols).enumerate() {
            let source = &planes.codes[row * cols..(row + 1) * cols];
            for (c, &p) in order.iter().enumerate() {
                dest[p * 4..p * 4 + 4].copy_from_slice(&source[c * 4..c * 4 + 4]);
            }
        }
        for (index, dest) in params.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let word = u32::from(planes.scales[index]) | (u32::from(planes.biases[index]) << 16);
            dest.copy_from_slice(&word.to_le_bytes());
        }
        Ok(Self {
            tensor,
            rows,
            cols,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        self.rows * self.cols + self.rows * (self.cols / GROUP) * 4
    }

    /// GPU dequantization into a BF16 [rows, cols] buffer (validation control).
    pub(super) fn encode_dequantize(
        &self,
        gpu: &mut Stream<'_>,
        output: &Buffer,
    ) {
        let cols4 = self.cols / 4;
        gpu.dispatch(
            "ink_a8_dequantize",
            &[binding(&self.tensor), binding(output)],
            [self.cols as u32, self.rows as u32],
            (cols4.div_ceil(64), self.rows),
            64,
        );
    }
}

/// Identity of a BF16 effective-weight buffer, used to find its codes.
pub(super) fn key(buffer: &Buffer) -> usize {
    std::ptr::from_ref::<ProtocolObject<dyn MTLBuffer>>(buffer).cast::<()>() as usize
}

/// The affine tensors of a loaded affine8 checkpoint.
pub(super) struct Affine8Weights {
    /// Attention and dynamic convolution projections, keyed by the BF16
    /// effective copy that prefill reads.
    dense: HashMap<usize, Affine8>,
    pub(super) embedding: Affine8,
    pub(super) head: Affine8,
    /// Diagnostic only: false runs the BF16 kernels on the effective copies.
    pub(super) decode: bool,
    /// Threads per matvec and readout threadgroup. Only the row-to-group
    /// assignment depends on them, never the reduction order.
    pub(super) threads: [usize; 2],
}

impl Affine8Weights {
    pub(super) fn new(
        embedding: Affine8,
        head: Affine8,
    ) -> Self {
        Self {
            dense: HashMap::new(),
            embedding,
            head,
            decode: true,
            threads: [AFFINE8_THREADS, READOUT_THREADS],
        }
    }

    pub(super) fn insert(
        &mut self,
        copy: &Buffer,
        tensor: Affine8,
    ) {
        self.dense.insert(key(copy), tensor);
    }

    /// The default threadgroup sizes.
    pub(super) fn new_threads() -> [usize; 2] {
        [AFFINE8_THREADS, READOUT_THREADS]
    }

    pub(super) fn tensor_for(
        &self,
        copy: &Buffer,
    ) -> Option<&Affine8> {
        self.dense.get(&key(copy))
    }

    pub(super) fn dense_bytes(&self) -> usize {
        self.dense.values().map(Affine8::bytes).sum()
    }

    pub(super) fn bytes(&self) -> usize {
        self.dense_bytes() + self.embedding.bytes() + self.head.bytes()
    }

    /// Load every affine tensor of an affine8 checkpoint. Returns the kernel
    /// tensors and the BF16 effective copies of the embedding (`vocabulary`
    /// rows) and of the head (`head_rows` rows, the copy padded likewise).
    pub(super) fn load(
        source: &Tensors,
        device: &Device,
        layers: &[Layer],
        vocabulary: usize,
        head_rows: usize,
    ) -> Result<(Self, Buffer, Buffer)> {
        let embed = Planes::read(source, "model.llm.embed.weight", [vocabulary, WIDTH])?;
        let embedding_copy = embed.effective_buffer(device, vocabulary, vocabulary)?;
        let embedding = Affine8::new(device, &embed, vocabulary)?;
        drop(embed);
        let unembed = Planes::read(source, "model.llm.unembed.weight", [vocabulary, WIDTH])?;
        let head_copy = unembed.effective_buffer(device, head_rows, head_rows)?;
        let head = Affine8::new(device, &unembed, head_rows)?;
        drop(unembed);
        let mut weights = Self::new(embedding, head);
        for (physical, layer) in layers.iter().enumerate() {
            let mut tensors = vec![
                ("attn.wq_du.weight", &layer.q, 3072, WIDTH),
                ("attn.wk_dv.weight", &layer.k, KV, WIDTH),
                ("attn.wv_dv.weight", &layer.v, KV, WIDTH),
                ("attn.wr_du.weight", &layer.r, HEADS * 16, WIDTH),
                ("attn.gate_attn.weight", &layer.attn_gate, 3072, WIDTH),
                ("attn.attn_sink_proj.weight", &layer.sink, HEADS, WIDTH),
                ("attn.wo_ud.weight", &layer.output, WIDTH, 3072),
            ];
            if let Some([key_proj, value_proj]) = &layer.dynamic {
                tensors.push(("attn.k_sconv.weight_proj", key_proj, 32, WIDTH));
                tensors.push(("attn.v_sconv.weight_proj", value_proj, 32, WIDTH));
            }
            for (name, copy, rows, cols) in tensors {
                let planes = Planes::read(source, &format!("model.llm.layers.{physical}.{name}"), [rows, cols])?;
                weights.insert(copy, Affine8::new(device, &planes, rows)?);
            }
        }
        Ok((weights, embedding_copy, head_copy))
    }
}

impl<'a> Stream<'a> {
    /// The affine codes behind a BF16 effective copy, when decode uses them.
    pub(super) fn affine8(
        &self,
        weight: &Buffer,
    ) -> Option<&'a Affine8> {
        let runtime: &'a Runtime = self.runtime;
        runtime.affine8.as_ref().filter(|weights| weights.decode).and_then(|weights| weights.dense.get(&key(weight)))
    }

    /// The affine head and embedding, when decode uses them.
    pub(super) fn affine8_vocabulary(&self) -> Option<&'a Affine8Weights> {
        let runtime: &'a Runtime = self.runtime;
        runtime.affine8.as_ref().filter(|weights| weights.decode)
    }

    /// Batched affine matvec for one to four live rows. Each row's output is
    /// independent of the batch and equals the BF16 kernels on the copy.
    pub(super) fn affine8_rows(
        &mut self,
        tensor: &Affine8,
        rows: &[(Binding<'_>, &Buffer)],
    ) {
        let count = rows.len();
        debug_assert!((1..=4).contains(&count));
        let kernel = ["ink_a8_matvec1", "ink_a8_matvec2", "ink_a8_matvec3", "ink_a8_matvec4"][count - 1];
        let mut buffers = [binding(&tensor.tensor); 9];
        for (index, &(input, output)) in rows.iter().enumerate() {
            buffers[1 + index] = input;
            buffers[1 + count + index] = binding(output);
        }
        let threads = self.runtime.affine8.as_ref().map_or(AFFINE8_THREADS, |weights| weights.threads[0]);
        let per_group = threads / lanes(tensor.rows);
        self.dispatch(
            kernel,
            &buffers[..1 + 2 * count],
            [tensor.cols as u32, tensor.rows as u32, count as u32],
            (tensor.rows.div_ceil(per_group), 1),
            threads,
        );
    }
}

/// Capped logits of the first `rows` hidden rows from the affine head. Each
/// row's reduction is independent of the number of rows.
pub(super) fn encode_readout(
    gpu: &mut Stream<'_>,
    weights: &Affine8Weights,
    hidden: &Buffer,
    logits: &Buffer,
    rows: usize,
    vocabulary: usize,
    padded: usize,
) {
    debug_assert!((1..=4).contains(&rows) && weights.head.rows == padded);
    let kernel = ["ink_a8_unembed1", "ink_a8_unembed2", "ink_a8_unembed3", "ink_a8_unembed4"][rows - 1];
    let head = &weights.head;
    let threads = weights.threads[1];
    gpu.dispatch(
        kernel,
        &[binding(hidden), binding(&head.tensor), binding(logits)],
        [head.cols as u32, vocabulary as u32, rows as u32, padded as u32],
        (vocabulary.div_ceil(threads / 16), 1),
        threads,
    );
}

const READOUT_THREADS: usize = 128;

/// Threads per affine matvec threadgroup. Only the row-to-group assignment
/// depends on it, never the reduction order.
const AFFINE8_THREADS: usize = 128;
