// SPDX-License-Identifier: Apache-2.0
//! Compare the packed graph with the pinned stock MLX Metal kernel.

use metal::{MTLDataType, MTLFunctionConstantValues};

use super::*;

/// Base key capacity, covered by the load-time source reserve. Non-source
/// requests and contexts up to this extent keep exactly this size.
pub(super) const MAX_KEYS: usize = CAPACITY;
/// Reference query tile, bounded by the explicit mask's size.
const TILE: usize = 256;
/// Default (mask-free) query tile. Each query row still visits the same key
/// blocks from key zero in the same order, and the blocks past its diagonal
/// are either skipped or contribute exact zeros, so outputs do not depend on
/// the tile; the larger tile fills the GPU.
const FAST_TILE: usize = 1024;
const MAX_QUERIES: usize = TILE + 63;
const MAX_FAST_QUERIES: usize = FAST_TILE + 63;

/// F32 per-head relative mask bytes for one query tile over `keys` keys.
pub(super) const fn mask_bytes(keys: usize) -> usize {
    HEADS * MAX_QUERIES * keys * size_of::<f32>()
}

/// F32 key and value staging bytes for `keys` keys.
pub(super) const fn key_value_bytes(keys: usize) -> usize {
    2 * keys * KV * size_of::<f32>()
}

/// Context-dependent global workspace bytes of the default path. The query,
/// output and sink buffers do not depend on the key capacity and are
/// excluded. The default path computes the relative bias in the attention
/// kernel; the explicit mask (`mask_bytes`) is allocated only when a
/// diagnostic requests the exact reference.
pub(super) const fn workspace_bytes(keys: usize) -> usize {
    key_value_bytes(keys)
}

/// Key capacity that serves a context. It never falls below the default so
/// contexts within the default extent keep the default buffers.
pub(super) fn capacity_for(context: usize) -> usize {
    context.clamp(MAX_KEYS, MAX_M5_CONTEXT)
}
const STEEL: &str = "steel_attention_float32_bq64_bk32_bd128_bv128_wm4_wn1_maskfloat32";
/// Stark derivative of STEEL that computes the relative bias in the kernel.
const RELATIVE: &str = "ink_global_relative_attention";
/// Default prefill: the same kernel with BF16 operands as stored and BF16 output.
const RELATIVE_BF16: &str = "ink_global_relative_attention_bf16";
const THIN: &str = "steel_gemm_thin_nax_nt_bfloat16_um2_un2_sm1_sn1_ks4";
const THIN_WIDE: &str = "steel_gemm_thin_nax_nt_bfloat16_um2_un2_sm1_sn2_ks2";

#[repr(C)]
struct AttnParams {
    dimensions: [i32; 6],
    scale: f32,
    tiles: [i32; 7],
    strides: [[i64; 3]; 4],
}

#[repr(C)]
struct ThinParams {
    dimensions: [i32; 8],
    batch_strides: [i64; 3],
    loop_geometry: [i32; 3],
}

pub(super) struct GlobalProbe {
    capacity: usize,
    pipelines: [Pipeline; 4],
    /// Default prefill: no mask tensor, BF16 operands. The exact reference
    /// keeps the F32 mask pipelines.
    relative: [Pipeline; 4],
    thin: [Pipeline; 4],
    thin_wide: [Pipeline; 4],
    query: Buffer,
    keys: Buffer,
    values: Buffer,
    output: Buffer,
    /// Explicit F32 relative mask, allocated on first use by the exact
    /// reference or the mask diagnostic and released on resize.
    mask: std::cell::RefCell<Option<Buffer>>,
    zero_sinks: Buffer,
}

impl GlobalProbe {
    pub(super) fn new(runtime: &mut Runtime) -> Result<Self> {
        register(
            runtime,
            include_str!("packed_global_probe.metal"),
            &[
                "ink_global_probe_query",
                "ink_global_probe_kv",
                "ink_global_probe_mask",
                "ink_global_probe_gate",
                "ink_global_probe_kv_bf16",
                "ink_global_probe_gate_bf16",
            ],
        )?;
        let device = &runtime.device;
        let library = packed_source_library(device)?;
        let specialize = |name, index: usize| -> Result<Pipeline> {
            // The mask-free kernel skips key blocks past the causal diagonal.
            // Their bias is -inf, so every output is unchanged; the reference
            // keeps the pinned full traversal.
            let causal = name == RELATIVE || name == RELATIVE_BF16;
            let constants = MTLFunctionConstantValues::new();
            for (id, value) in
                [(200, index & 1 != 0), (201, index & 2 != 0), (300, true), (301, causal), (302, true), (303, false)]
            {
                constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::Bool, id);
            }
            constants.set_constant_value_type_at_index(NonNull::from(&-1i32).cast(), MTLDataType::Int, 304);
            let function = library
                .new_function_with_name_constant_values(name, &constants)
                .map_err(|_| "cannot specialize pinned MLX global kernel")?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|_| "cannot create pinned MLX global pipeline")
        };
        let thin = |name, index: usize| -> Result<Pipeline> {
            let constants = MTLFunctionConstantValues::new();
            for (id, value) in [(100, false), (110, false), (200, index & 1 != 0), (201, index & 2 != 0), (202, true)] {
                constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::Bool, id);
            }
            let function = library
                .new_function_with_name_constant_values(name, &constants)
                .map_err(|_| "cannot specialize pinned MLX thin projection")?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|_| "cannot create pinned MLX thin projection pipeline")
        };
        Ok(Self {
            capacity: MAX_KEYS,
            pipelines: [specialize(STEEL, 0)?, specialize(STEEL, 1)?, specialize(STEEL, 2)?, specialize(STEEL, 3)?],
            relative: [
                specialize(RELATIVE_BF16, 0)?,
                specialize(RELATIVE_BF16, 1)?,
                specialize(RELATIVE_BF16, 2)?,
                specialize(RELATIVE_BF16, 3)?,
            ],
            thin: [thin(THIN, 0)?, thin(THIN, 1)?, thin(THIN, 2)?, thin(THIN, 3)?],
            thin_wide: [thin(THIN_WIDE, 0)?, thin(THIN_WIDE, 1)?, thin(THIN_WIDE, 2)?, thin(THIN_WIDE, 3)?],
            query: zeroed(device, MAX_FAST_QUERIES * QUERY * 4)?,
            keys: zeroed(device, MAX_KEYS * KV * 4)?,
            values: zeroed(device, MAX_KEYS * KV * 4)?,
            output: zeroed(device, MAX_FAST_QUERIES * QUERY * 4)?,
            mask: std::cell::RefCell::new(None),
            zero_sinks: zeroed(device, HEADS * 4)?,
        })
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// The explicit mask for the exact reference, allocated at the current
    /// key capacity on first use. Diagnostic paths only.
    fn reference_mask(
        &self,
        device: &Device,
    ) -> Buffer {
        let mut mask = self.mask.borrow_mut();
        let bytes = mask_bytes(self.capacity);
        if mask.as_ref().is_none_or(|buffer| buffer.length() < bytes) {
            *mask = Some(zeroed(device, bytes).expect("Inkling reference global mask allocation"));
        }
        mask.as_ref().expect("allocated above").clone()
    }

    /// Reallocate the key-extent buffers for the context-sized workspace.
    /// The previous buffers are released before the replacements are
    /// allocated, so the peak never exceeds the admitted size. If an
    /// allocation fails the capacity stays zero and no context is ready
    /// until a later resize succeeds. Kernels, dispatch geometry and
    /// arithmetic do not depend on capacity.
    pub(super) fn resize(
        &mut self,
        device: &Device,
        keys: usize,
    ) -> Result<()> {
        if keys == self.capacity {
            return Ok(());
        }
        if !(MAX_KEYS..=MAX_M5_CONTEXT).contains(&keys) || mask_bytes(keys) > device.max_buffer_length() {
            return Err("Inkling source global workspace exceeds the supported key capacity");
        }
        self.capacity = 0;
        let placeholder = || zeroed(device, size_of::<f32>());
        self.mask.replace(None);
        self.keys = placeholder()?;
        self.values = placeholder()?;
        self.keys = zeroed(device, keys * KV * 4)?;
        self.values = zeroed(device, keys * KV * 4)?;
        self.capacity = keys;
        Ok(())
    }

    pub(super) fn project_thin(
        &self,
        gpu: &mut Stream<'_>,
        input: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        width: usize,
        columns: usize,
        rows: usize,
    ) {
        let Some(encoder) = &gpu.encoder else {
            return;
        };
        let (pipeline, tile_columns, k_step) = if rows >= 2048 {
            (&self.thin_wide, 64, 128)
        } else {
            (&self.thin, 32, 256)
        };
        let alignments =
            usize::from(rows.is_multiple_of(32)) | (usize::from(columns.is_multiple_of(tile_columns)) << 1);
        let cols = columns.div_ceil(tile_columns);
        let tiles = rows.div_ceil(32);
        let params = ThinParams {
            dimensions: [
                rows as i32,
                columns as i32,
                width as i32,
                width as i32,
                width as i32,
                columns as i32,
                cols as i32,
                tiles as i32,
            ],
            batch_strides: [0; 3],
            loop_geometry: [0, width.div_ceil(k_step) as i32, 0],
        };
        encoder.set_compute_pipeline_state(&pipeline[alignments]);
        encoder.set_buffer(Some(input), 0, 0);
        encoder.set_buffer(Some(weight), 0, 1);
        encoder.set_buffer(Some(output), 0, 3);
        encoder.set_bytes(NonNull::from(&params).cast(), size_of::<ThinParams>(), 4);
        encoder.dispatch_threadgroups(
            MTLSize {
                width: cols * tiles,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        gpu.dispatches += 1;
    }

    pub(super) fn encode(
        &self,
        gpu: &mut Stream<'_>,
        packed: &PackedPrefill,
        visit: &Visit,
        start: usize,
        rows: usize,
    ) {
        let extent = start + rows;
        assert!(extent <= self.capacity && extent <= visit.capacity);
        let reference = gpu.runtime.exact_prefill || gpu.runtime.global_mask_reference;
        gpu.phase(PrefillPhase::GlobalPacking);
        gpu.one(
            if reference {
                "ink_global_probe_kv"
            } else {
                "ink_global_probe_kv_bf16"
            },
            &[&visit.keys, &visit.values, &packed.keys, &packed.values, &self.keys, &self.values],
            [start as u32, extent as u32],
            (KV / 128, extent),
            128,
        );
        let tile = if reference {
            TILE
        } else {
            FAST_TILE
        };
        let mut offset = 0;
        while offset < rows {
            let proposed = (offset + tile).min(rows);
            let end = if rows - proposed < 64 {
                rows
            } else {
                proposed
            };
            let queries = end - offset;
            let keys = start + end;
            if reference {
                gpu.phase(PrefillPhase::GlobalPacking);
                gpu.one(
                    "ink_global_probe_query",
                    &[&packed.q, &self.query],
                    [queries as u32, offset as u32],
                    (QUERY / 128, queries),
                    128,
                );
            }
            let relative = [queries as u32, keys as u32, offset as u32, 0];
            if reference {
                let mask = self.reference_mask(&gpu.runtime.device);
                gpu.phase(PrefillPhase::GlobalMask);
                gpu.one(
                    "ink_global_probe_mask",
                    &[&packed.table, &packed.sink, &mask],
                    relative,
                    (keys.div_ceil(128), HEADS * queries),
                    128,
                );
            }
            gpu.phase(PrefillPhase::GlobalAttention);
            self.encode_matrix(gpu, packed, extent, offset, queries, keys, relative);
            gpu.phase(PrefillPhase::GlobalGate);
            gpu.one(
                if reference {
                    "ink_global_probe_gate"
                } else {
                    "ink_global_probe_gate_bf16"
                },
                &[&self.output, &packed.gate, &packed.attended],
                [queries as u32, offset as u32],
                (QUERY / 128, queries),
                128,
            );
            offset = end;
        }
    }

    fn encode_matrix(
        &self,
        gpu: &mut Stream<'_>,
        packed: &PackedPrefill,
        extent: usize,
        offset: usize,
        queries: usize,
        keys: usize,
        relative: [u32; 4],
    ) {
        let Some(encoder) = &gpu.encoder else {
            return;
        };
        let reference = gpu.runtime.exact_prefill || gpu.runtime.global_mask_reference;
        let alignments = usize::from(queries.is_multiple_of(64)) | (usize::from(keys.is_multiple_of(32)) << 1);
        let kv_strides = [(extent * KV) as i64, (extent * 128) as i64, 128];
        let params = AttnParams {
            dimensions: [1, HEADS as i32, 128, queries as i32, keys as i32, 3],
            scale: 0.088_388_348,
            tiles: [
                queries.div_ceil(64) as i32,
                keys.div_ceil(32) as i32,
                (queries / 64) as i32,
                (keys / 32) as i32,
                (queries % 64) as i32,
                (keys % 32) as i32,
                (keys - queries) as i32,
            ],
            strides: [
                // The default path reads queries in place from the packed
                // [row, head x 128] layout; the reference packs them head-major.
                if reference {
                    [(queries * QUERY) as i64, (queries * 128) as i64, 128]
                } else {
                    [(queries * QUERY) as i64, 128, QUERY as i64]
                },
                kv_strides,
                kv_strides,
                [(queries * QUERY) as i64, 128, QUERY as i64],
            ],
        };
        let mask_strides = [(HEADS * queries * keys) as i64, (queries * keys) as i64, keys as i64];
        for (index, buffer) in [&self.query, &self.keys, &self.values, &self.output].into_iter().enumerate() {
            encoder.set_buffer(Some(buffer), 0, index);
        }
        if !reference {
            encoder.set_buffer(Some(&packed.q), offset * QUERY * size_of::<u16>(), 0);
        }
        encoder.set_bytes(NonNull::from(&params).cast(), size_of::<AttnParams>(), 4);
        encoder.set_buffer(Some(&self.zero_sinks), 0, 7);
        if reference {
            encoder.set_compute_pipeline_state(&self.pipelines[alignments]);
            encoder.set_bytes(NonNull::from(&mask_strides).cast(), size_of_val(&mask_strides), 5);
            encoder.set_buffer(Some(&self.reference_mask(&gpu.runtime.device)), 0, 6);
        } else {
            // The same F32 bias values, computed from the table and sink;
            // BF16 operands as stored.
            encoder.set_compute_pipeline_state(&self.relative[alignments]);
            encoder.set_bytes(NonNull::from(&relative).cast(), size_of_val(&relative), 5);
            encoder.set_buffer(Some(&packed.table), 0, 6);
            encoder.set_buffer(Some(&packed.sink), 0, 9);
        }
        encoder.dispatch_threadgroups(
            MTLSize {
                width: queries.div_ceil(64),
                height: HEADS,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            },
        );
        gpu.dispatches += 1;
    }
}
