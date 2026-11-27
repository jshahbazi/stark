// SPDX-License-Identifier: Apache-2.0
//! Native diagnostic entry for qualifying the pinned STQ artifact.

use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    os::fd::AsRawFd,
    ptr::{self, NonNull},
};

use super::{
    ContextMemory, ContextSelection, Session, VisionGeometry, select_context,
    stq_layout::{StqFormat, StqProjection},
    stq_source::{StqDtype, StqSource, StqTensorLocation},
    stq_trunk::{StqProjectionUse, StqRawTransform, StqTrunkWeight},
};

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct TensorSlice {
    fd: c_int,
    offset: u64,
    size: u64,
}

impl From<&StqTensorLocation<'_>> for TensorSlice {
    fn from(location: &StqTensorLocation<'_>) -> Self {
        Self {
            fd: location.file.as_raw_fd(),
            offset: location.offset,
            size: location.byte_len as u64,
        }
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct ProjectionSource {
    inputs: u32,
    outputs: u32,
    rate: u32,
    trellis: TensorSlice,
    suh: TensorSlice,
    svh: TensorSlice,
}

impl From<&StqProjection<'_>> for ProjectionSource {
    fn from(projection: &StqProjection<'_>) -> Self {
        Self {
            inputs: projection.inputs,
            outputs: projection.outputs,
            rate: match projection.format {
                StqFormat::Pair1K4 | StqFormat::Spec1K4 => 4,
                StqFormat::Spec1K6 => 6,
            },
            trellis: (&projection.trellis).into(),
            suh: (&projection.input_scale).into(),
            svh: (&projection.output_scale).into(),
        }
    }
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum WeightUse {
    Raw,
    Widen,
    Linear,
    AbsorbedKey,
    AbsorbedValue,
}

struct Binding {
    alias: CString,
    kind: WeightUse,
    dtype: u32,
    shape: Vec<u32>,
    raw: Vec<TensorSlice>,
    projection: ProjectionSource,
}

#[repr(C)]
struct WeightSource {
    alias: *const c_char,
    kind: WeightUse,
    dtype: u32,
    rank: u32,
    shape: [u32; 3],
    pieces: *const TensorSlice,
    piece_count: usize,
    projection: ProjectionSource,
}

impl Binding {
    fn new(
        alias: &str,
        weight: &StqTrunkWeight<'_>,
    ) -> Result<Self, String> {
        let mut binding = Self {
            alias: CString::new(alias).map_err(|_| "invalid STQ alias")?,
            kind: WeightUse::Raw,
            dtype: 0,
            shape: Vec::new(),
            raw: Vec::new(),
            projection: ProjectionSource::default(),
        };
        let (shape, dtype, transform) = match weight {
            StqTrunkWeight::Tensor {
                source,
                transform,
            } => {
                binding.raw.push((&source.location).into());
                (source.shape, source.dtype, *transform)
            },
            StqTrunkWeight::Concatenate {
                sources,
                transform,
                shape,
            } => {
                binding.raw.extend(sources.iter().map(|s| TensorSlice::from(&s.location)));
                (*shape, StqDtype::BF16, *transform)
            },
            StqTrunkWeight::EngramGateNorms {
                streams,
            } => {
                for stream in streams {
                    binding.raw.push((&stream.key_norm.location).into());
                    binding.raw.push((&stream.query_norm.location).into());
                }
                (&[4, 2, 3072][..], StqDtype::BF16, StqRawTransform::WidenToF32)
            },
            StqTrunkWeight::Projection {
                source,
                use_as,
            } => {
                let p = &source.projection;
                binding.projection = p.into();
                match *use_as {
                    StqProjectionUse::Embedding | StqProjectionUse::Linear => {
                        binding.kind = WeightUse::Linear;
                        binding.shape = vec![p.inputs, p.outputs];
                    },
                    StqProjectionUse::AbsorbedKey {
                        heads,
                    } => {
                        binding.kind = WeightUse::AbsorbedKey;
                        binding.dtype = 30;
                        binding.shape = vec![192, 512, heads];
                    },
                    StqProjectionUse::AbsorbedValue {
                        heads,
                    } => {
                        binding.kind = WeightUse::AbsorbedValue;
                        binding.dtype = 30;
                        binding.shape = vec![512, 256, heads];
                    },
                }
                return Ok(binding);
            },
        };
        binding.shape.extend(shape.iter().rev().copied());
        match (dtype, transform) {
            (StqDtype::F32, StqRawTransform::AsStored) => {},
            (StqDtype::BF16, StqRawTransform::AsStored) => binding.dtype = 30,
            (StqDtype::BF16, StqRawTransform::WidenToF32) => binding.kind = WeightUse::Widen,
            _ => return Err(format!("invalid STQ raw dtype for {alias}")),
        }
        Ok(binding)
    }
}

#[derive(Clone, Copy, Default)]
pub struct StqOptions {
    pub chunk: u32,
    pub context_limit: u32,
    pub profile: bool,
    pub round_bf16: bool,
    pub pair1_decode: bool,
    pub resident: bool,
    pub tensor_mla: bool,
    pub gpu_verify_plan: bool,
    pub register_mla: bool,
    pub fused_finish_reduce: bool,
    pub split_spec1: bool,
    pub verify_register_mla: bool,
    pub cache_engram_pages: bool,
    pub wide_attention: bool,
    pub wide_routed: bool,
    pub draft_register_mla: bool,
    pub wide_ffn: bool,
    pub fast_verify: bool,
    pub fused_attn_hc: bool,
    pub batch_draft: bool,
    pub register_dense_io: bool,
}

unsafe extern "C" {
    fn stark_hy4_open_stq_diagnostic(
        weights: *const WeightSource,
        weight_count: usize,
        experts: *const ProjectionSource,
        expert_count: usize,
        tables: *const TensorSlice,
        token_map: *const i32,
        map_len: usize,
        vision_weights: *const TensorSlice,
        vision_count: usize,
        chunk: c_int,
        profile: c_int,
        round_bf16: c_int,
        pair1_decode: c_int,
        resident: c_int,
        tuning: u32,
        select_context: unsafe extern "C" fn(*const ContextMemory, *mut c_void) -> u32,
        sizing: *mut c_void,
        session: *mut *mut c_void,
        vision: *mut VisionGeometry,
        detail: *mut c_char,
        detail_len: usize,
    ) -> c_int;
}

impl Session {
    /// Open the pinned artifact for native qualification.
    ///
    /// # Safety
    /// The validated files must remain immutable for this session's lifetime,
    /// including all GPU work. The native runtime maps the original packed pages.
    pub unsafe fn load_stq_diagnostic(
        source: &StqSource,
        options: StqOptions,
    ) -> Result<Self, String> {
        let trunk = source.stq_trunk().map_err(|e| e.to_string())?;
        let dfly = source.stq_dfly().map_err(|e| e.to_string())?;
        let bindings: Vec<_> = trunk
            .weights()
            .chain(dfly.weights())
            .map(|(alias, weight)| Binding::new(alias, weight))
            .collect::<Result<_, _>>()?;
        let weights: Vec<_> = bindings
            .iter()
            .map(|b| {
                let mut shape = [0; 3];
                shape[..b.shape.len()].copy_from_slice(&b.shape);
                WeightSource {
                    alias: b.alias.as_ptr(),
                    kind: b.kind,
                    dtype: b.dtype,
                    rank: b.shape.len() as u32,
                    shape,
                    pieces: b.raw.as_ptr(),
                    piece_count: b.raw.len(),
                    projection: b.projection,
                }
            })
            .collect();
        let mut experts = Vec::with_capacity(123);
        for block in 1..42 {
            experts.extend(
                trunk.routed_experts(block).ok_or("missing STQ expert block")?.iter().map(ProjectionSource::from),
            );
        }
        let mut tables = [TensorSlice::default(); 4];
        for (table, block) in tables.iter_mut().zip([1, 11, 21, 31]) {
            *table = (&source.stq_engram_table(block).ok_or("missing STQ Engram table")?.location).into();
        }
        let token_map = source.stq_engram_token_map().map_err(|e| e.to_string())?;
        let mut vision_weights = Vec::with_capacity(441);
        let mut vision_names: Vec<_> = source.tensor_names().filter(|name| name.starts_with("vit.")).collect();
        vision_names.sort_unstable();
        for name in vision_names {
            let tensor = source.tensor(name).ok_or("missing STQ vision tensor")?;
            if tensor.dtype != StqDtype::BF16 {
                return Err("unexpected STQ vision dtype".into());
            }
            let location = source.tensor_location(name).ok_or("missing STQ vision slice")?;
            if location.shard_name != "model-globals.safetensors" {
                return Err("unexpected STQ vision shard".into());
            }
            vision_weights.push(TensorSlice::from(&location));
        }
        source.verify_unchanged().map_err(|e| e.to_string())?;
        let mut selection = ContextSelection {
            limit: Some(options.context_limit),
            pool_size: None,
            sizing: None,
        };
        let mut inner = ptr::null_mut();
        let mut vision = VisionGeometry::default();
        let mut detail = [0; 256];
        // Only the mapped page ranges and duplicated table descriptors survive
        // open. All temporary alias strings and descriptors remain valid here.
        let status = unsafe {
            stark_hy4_open_stq_diagnostic(
                weights.as_ptr(),
                weights.len(),
                experts.as_ptr(),
                experts.len(),
                tables.as_ptr(),
                token_map.as_ptr(),
                token_map.len(),
                vision_weights.as_ptr(),
                vision_weights.len(),
                options.chunk as c_int,
                c_int::from(options.profile),
                c_int::from(options.round_bf16),
                c_int::from(options.pair1_decode),
                c_int::from(options.resident),
                u32::from(options.tensor_mla)
                    | (u32::from(options.gpu_verify_plan) << 1)
                    | (u32::from(options.register_mla) << 2)
                    | (u32::from(options.fused_finish_reduce) << 3)
                    | (u32::from(options.split_spec1) << 4)
                    | (u32::from(options.verify_register_mla) << 5)
                    | (u32::from(options.cache_engram_pages) << 6)
                    | (u32::from(options.wide_attention) << 7)
                    | (u32::from(options.wide_routed) << 8)
                    | (u32::from(options.draft_register_mla) << 9)
                    | (u32::from(options.wide_ffn) << 10)
                    | (u32::from(options.fast_verify) << 11)
                    | (u32::from(options.fused_attn_hc) << 12)
                    | (u32::from(options.batch_draft) << 13)
                    | (u32::from(options.register_dense_io) << 14),
                select_context,
                ptr::from_mut(&mut selection).cast(),
                &mut inner,
                &mut vision,
                detail.as_mut_ptr(),
                detail.len(),
            )
        };
        if status != 0 {
            return Err(unsafe { CStr::from_ptr(detail.as_ptr()) }.to_string_lossy().into_owned());
        }
        let session = Self {
            inner: NonNull::new(inner).ok_or("invalid STQ native session")?,
            vision,
            mapped_experts: true,
            draft_head_q5: false,
            arena_bytes: 0,
            sizing: selection.sizing.expect("successful STQ open selected a context"),
            pool_size: selection.pool_size.expect("successful STQ open selected a pool size"),
            max_drafts: dfly.drafts as usize,
        };
        source.verify_unchanged().map_err(|e| e.to_string())?;
        Ok(session)
    }
}
