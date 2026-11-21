// SPDX-License-Identifier: Apache-2.0
//! Compare the packed graph with the pinned stock MLX Metal kernel.

use metal::{MTLDataType, MTLFunctionConstantValues};

use super::*;

pub(super) const MAX_KEYS: usize = 4352;
const TILE: usize = 256;
const MAX_QUERIES: usize = TILE + 63;
const STEEL: &str = "steel_attention_float32_bq64_bk32_bd128_bv128_wm4_wn1_maskfloat32";
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
    pipelines: [Pipeline; 4],
    thin: [Pipeline; 4],
    thin_wide: [Pipeline; 4],
    query: Buffer,
    keys: Buffer,
    values: Buffer,
    output: Buffer,
    mask: Buffer,
    zero_sinks: Buffer,
}

impl GlobalProbe {
    pub(super) fn new(runtime: &mut Runtime) -> Result<Self> {
        register(
            runtime,
            include_str!("packed_global_probe.metal"),
            &["ink_global_probe_query", "ink_global_probe_kv", "ink_global_probe_mask", "ink_global_probe_gate"],
        )?;
        let device = &runtime.device;
        let library = packed_source_library(device)?;
        let specialize = |index: usize| -> Result<Pipeline> {
            let constants = MTLFunctionConstantValues::new();
            for (id, value) in
                [(200, index & 1 != 0), (201, index & 2 != 0), (300, true), (301, false), (302, true), (303, false)]
            {
                constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::Bool, id);
            }
            constants.set_constant_value_type_at_index(NonNull::from(&-1i32).cast(), MTLDataType::Int, 304);
            let function = library
                .new_function_with_name_constant_values(STEEL, &constants)
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
            pipelines: [specialize(0)?, specialize(1)?, specialize(2)?, specialize(3)?],
            thin: [thin(THIN, 0)?, thin(THIN, 1)?, thin(THIN, 2)?, thin(THIN, 3)?],
            thin_wide: [thin(THIN_WIDE, 0)?, thin(THIN_WIDE, 1)?, thin(THIN_WIDE, 2)?, thin(THIN_WIDE, 3)?],
            query: zeroed(device, MAX_QUERIES * QUERY * 4)?,
            keys: zeroed(device, MAX_KEYS * KV * 4)?,
            values: zeroed(device, MAX_KEYS * KV * 4)?,
            output: zeroed(device, MAX_QUERIES * QUERY * 4)?,
            mask: zeroed(device, HEADS * MAX_QUERIES * MAX_KEYS * 4)?,
            zero_sinks: zeroed(device, HEADS * 4)?,
        })
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
        let alignments = usize::from(rows % 32 == 0) | (usize::from(columns % tile_columns == 0) << 1);
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
        assert!(extent <= MAX_KEYS && extent <= visit.capacity);
        gpu.one(
            "ink_global_probe_kv",
            &[&visit.keys, &visit.values, &packed.keys, &packed.values, &self.keys, &self.values],
            [start as u32, extent as u32],
            (KV / 128, extent),
            128,
        );
        let mut offset = 0;
        while offset < rows {
            let proposed = (offset + TILE).min(rows);
            let end = if rows - proposed < 64 {
                rows
            } else {
                proposed
            };
            let queries = end - offset;
            let keys = start + end;
            gpu.one(
                "ink_global_probe_query",
                &[&packed.q, &self.query],
                [queries as u32, offset as u32],
                (QUERY / 128, queries),
                128,
            );
            gpu.one(
                "ink_global_probe_mask",
                &[&packed.table, &packed.sink, &self.mask],
                [queries as u32, keys as u32, offset as u32, 0],
                (keys.div_ceil(128), HEADS * queries),
                128,
            );
            self.encode_matrix(gpu, extent, queries, keys);
            gpu.one(
                "ink_global_probe_gate",
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
        extent: usize,
        queries: usize,
        keys: usize,
    ) {
        let Some(encoder) = &gpu.encoder else {
            return;
        };
        let alignments = usize::from(queries % 64 == 0) | (usize::from(keys % 32 == 0) << 1);
        let kv_strides = [(extent * KV) as i64, (extent * 128) as i64, 128];
        let params = AttnParams {
            dimensions: [1, HEADS as i32, 128, queries as i32, keys as i32, 3],
            scale: 0.08838834764831845,
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
                [(queries * QUERY) as i64, (queries * 128) as i64, 128],
                kv_strides,
                kv_strides,
                [(queries * QUERY) as i64, 128, QUERY as i64],
            ],
        };
        let mask_strides = [(HEADS * queries * keys) as i64, (queries * keys) as i64, keys as i64];
        encoder.set_compute_pipeline_state(&self.pipelines[alignments]);
        for (index, buffer) in [&self.query, &self.keys, &self.values, &self.output].into_iter().enumerate() {
            encoder.set_buffer(Some(buffer), 0, index);
        }
        encoder.set_bytes(NonNull::from(&params).cast(), size_of::<AttnParams>(), 4);
        encoder.set_bytes(NonNull::from(&mask_strides).cast(), size_of_val(&mask_strides), 5);
        encoder.set_buffer(Some(&self.mask), 0, 6);
        encoder.set_buffer(Some(&self.zero_sinks), 0, 7);
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
