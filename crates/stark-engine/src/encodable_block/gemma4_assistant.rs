//! Official small Gemma 4 companion draft step. Architecture provenance:
//! Transformers 5.15.1 modeling_gemma4_assistant.py (Apache-2.0, Copyright
//! 2026 HuggingFace Team), SHA-256
//! a77e67673ce7d2248294e361c3c074c29d8d6a092655d7b27b95fc9777a2d0d1.
//! This component does not advertise MTP or implement target verification.
use std::sync::Arc;

use derive_more::Debug;
use half::bf16;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
        gpu_types::ActivationType,
        kernel::{GatedActMul, GatedActMulSettings, Kernels, MaterializedBF16AddScaleKernel, VisionGeluBF16Kernel},
    },
    config::{
        normalization::NormalizationConfig,
        rope::AnyRoPEConfig,
        token_mixer::attention::AttentionConfig,
        weight_matrix::{AnyWeightMatrixSpec, Layout},
    },
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError, ReferenceBF16Matmul},
        mixer::attention::{
            Attention, AttentionNewError, CommittedAttentionKV, reference::ReferenceBF16Attention,
            rope::PrecalculatedRoPE,
        },
        mlp::{Mlp, MlpBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
    },
    parameters::{NumericalProfile, ParameterLoaderError, ParameterTree},
};

const DIM: u32 = 256;
const VOCAB: u32 = 262144;
const CENTROIDS: u32 = 2048;
const TOP_K: usize = 32;

/// Pinned-source MLP reductions, GELU and a separate BF16 product are confined
/// to the assistant. Its other linears retain their legacy implementation.
struct AssistantSourceGeluMlp<B: Backend> {
    gelu: <B::Kernels as Kernels>::VisionGeluBF16Kernel,
    multiply: GatedActMul<B>,
    reference: ReferenceBF16Matmul<B>,
    up_weights: Arc<B::GlobalBuffer>,
    down_weights: Arc<B::GlobalBuffer>,
}

impl<B: Backend> AssistantSourceGeluMlp<B> {
    fn project_and_gate(
        &self,
        input: B::ScratchBuffer,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer, B::ScratchBuffer), B::Error> {
        if rows != 1 {
            return Err(crate::backends::common::InvalidInput("assistant MLP requires one query").into());
        }
        let fused = self.reference.encode(&input, 0, &*self.up_weights, rows, 4096, DIM, commands)?;
        let bytes = 2048 * size_of::<bf16>();
        if fused.size() != bytes * 2 {
            return Err(crate::backends::common::InvalidInput("invalid assistant fused MLP shape").into());
        }
        let mut activated = commands.allocate_scratch(bytes)?;
        commands.encode_copy((&fused).subrange(bytes..bytes * 2), &mut activated);
        self.gelu.encode(&mut activated, 2048, 1, commands);
        let mut product = commands.allocate_scratch(bytes)?;
        self.multiply.encode_fp(
            &activated,
            Some(&fused),
            &mut product,
            None::<&B::ScratchBuffer>,
            2048,
            1,
            0,
            4096,
            ActivationType::IDENTITY,
            commands,
        );
        Ok((fused, activated, product))
    }
}

impl<B: Backend> Mlp<B> for AssistantSourceGeluMlp<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let (_fused, _activated, product) = self.project_and_gate(input, rows, commands)?;
        self.reference.encode(&product, 0, &*self.down_weights, rows, DIM, 2048, commands)
    }

    #[cfg(test)]
    fn trace_dense(
        &self,
        input: B::ScratchBuffer,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Option<Vec<(String, B::ScratchBuffer)>>, B::Error> {
        let (fused, activated, product) = self.project_and_gate(input, rows, commands)?;
        // The independently qualified primitive candidates are now the actual
        // MLP operators; retain their names for the separate source trace.
        let mut reference_up = commands.allocate_scratch(fused.size())?;
        commands.encode_copy(&fused, &mut reference_up);
        let mut act_copy = commands.allocate_scratch(activated.size())?;
        let mut product_copy = commands.allocate_scratch(product.size())?;
        let mut product_candidate = commands.allocate_scratch(product.size())?;
        commands.encode_copy(&activated, &mut act_copy);
        commands.encode_copy(&product, &mut product_copy);
        commands.encode_copy(&product, &mut product_candidate);
        let down = self.reference.encode(&product, 0, &*self.down_weights, rows, DIM, 2048, commands)?;
        let mut reference_down = commands.allocate_scratch(down.size())?;
        commands.encode_copy(&down, &mut reference_down);
        let mut down_copy = commands.allocate_scratch(down.size())?;
        commands.encode_copy(&down, &mut down_copy);
        Ok(Some(vec![
            ("mlp.fused_up".into(), fused),
            ("mlp.activation".into(), activated),
            ("mlp.product".into(), product_copy),
            ("mlp.down".into(), down),
            ("mlp.reference_activation_candidate".into(), act_copy),
            ("mlp.reference_product_candidate".into(), product_candidate),
            ("mlp.reference_down_candidate".into(), down_copy),
            ("mlp.reference_up_linear_candidate".into(), reference_up),
            ("mlp.reference_down_linear_candidate".into(), reference_down),
        ]))
    }
}

#[derive(Clone, Copy)]
pub enum Gemma4AssistantProfile {
    E2B,
    E4B,
}
impl Gemma4AssistantProfile {
    pub fn backbone_dim(self) -> u32 {
        match self {
            Self::E2B => 1536,
            Self::E4B => 2560,
        }
    }
    fn kv_heads(self) -> u32 {
        match self {
            Self::E2B => 1,
            Self::E4B => 2,
        }
    }
}

#[derive(Debug, Error)]
pub enum Gemma4AssistantError<B: Backend> {
    #[error("invalid Gemma assistant graph, inputs or output")]
    Invalid,
    #[error("Gemma assistant backend failed")]
    Backend(#[source] B::Error),
    #[error("Gemma assistant parameter loading failed")]
    Parameters(#[from] ParameterLoaderError<B>),
    #[error("Gemma assistant normalization loading failed")]
    Norm(#[from] NormalizationNewError<B>),
    #[error("Gemma assistant attention loading failed")]
    Attention(#[from] AttentionNewError<B>),
    #[error("Gemma assistant linear loading failed")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Gemma assistant MLP loading failed")]
    Mlp(#[from] MlpBlockError<B>),
    #[error("Gemma assistant static configuration failed")]
    Config(#[from] serde_json::Error),
}

struct AssistantLayer<B: Backend> {
    pre_attention: Normalization<B>,
    attention: Attention<B>,
    eager: ReferenceBF16Attention<B>,
    post_attention: Normalization<B>,
    pre_mlp: Normalization<B>,
    mlp: Box<dyn Mlp<B>>,
    post_mlp: Normalization<B>,
    scalar: f32,
    rope: AnyRoPEConfig,
}

pub struct Gemma4Assistant<B: Backend> {
    profile: Gemma4AssistantProfile,
    target_identity: Option<Arc<()>>,
    pre_projection: Box<dyn Linear<B>>,
    post_projection: Box<dyn Linear<B>>,
    layers: Box<[AssistantLayer<B>]>,
    output_norm: Normalization<B>,
    centroids: Box<dyn Linear<B>>,
    readout: Box<dyn Linear<B>>,
    ordering: Box<[u32]>,
    add: <B::Kernels as Kernels>::MaterializedBF16AddScaleKernel,
}

/// Device-owned outputs, with no Debug or serialization of activations/logits.
pub struct Gemma4AssistantStep<B: Backend> {
    pub feedback: B::ScratchBuffer,
    pub centroids: B::ScratchBuffer,
    pub logits: B::ScratchBuffer,
}

impl<B: Backend> Gemma4Assistant<B> {
    pub fn numerical_policy(&self) -> &'static str {
        "gemma4-assistant-eager-bf16-source-mlp-v1"
    }

    pub fn backbone_dim(&self) -> u32 {
        self.profile.backbone_dim()
    }
    pub(crate) fn bind_target(
        mut self,
        identity: Arc<()>,
    ) -> Self {
        self.target_identity = Some(identity);
        self
    }
    pub(crate) fn matches_target(
        &self,
        identity: &Arc<()>,
    ) -> bool {
        self.target_identity.as_ref().is_some_and(|bound| Arc::ptr_eq(bound, identity))
    }
    /// The engine must first admit the complete companion and bind it to the
    /// actual selected target descriptor pair. This low-level constructor cannot
    /// grant capability from a companion filename or declared config hash.
    pub fn new(
        profile: Gemma4AssistantProfile,
        tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, Gemma4AssistantError<B>> {
        if tree.numerical_profile() != NumericalProfile::Legacy {
            return Err(Gemma4AssistantError::Invalid);
        }
        let norm_config: NormalizationConfig = serde_json::from_value(serde_json::json!({
            "epsilon":1e-6,"scale_offset":null,"upcast_mode":"full_layer",
            "subtract_mean":false,"has_scale":true,"has_biases":false
        }))?;
        let norm = |name: &str, width| {
            Normalization::new(
                width,
                None,
                ShortcutMode::None,
                PostLayerScalar::None,
                DataType::BF16,
                &norm_config,
                &tree.subtree(name),
                context,
            )
        };
        let full_matrix = |name: &str| -> Result<(), Gemma4AssistantError<B>> {
            if !matches!(tree.subtree(name).subtree("weights").metadata::<AnyWeightMatrixSpec>("spec")?,
                AnyWeightMatrixSpec::FullPrecisionSpec(spec) if spec.layout == Layout::OutputInput)
            {
                return Err(Gemma4AssistantError::Invalid);
            }
            Ok(())
        };
        let linear = |name: &str, input, output| -> Result<_, Gemma4AssistantError<B>> {
            full_matrix(name)?;
            Ok(<dyn Linear<B>>::new(input, [output], false, context, DataType::BF16, &tree.subtree(name))?)
        };
        let mut layers = Vec::new();
        for index in 0..4 {
            let base = format!("layers.{index}");
            for projection in
                ["mixer.qkvg_projection", "mixer.out_projection", "mlp.up_projection", "mlp.down_projection"]
            {
                full_matrix(&format!("{base}.{projection}"))?;
            }
            let global = index == 3;
            let width = if global {
                512
            } else {
                256
            };
            let rope: AnyRoPEConfig = serde_json::from_value(if global {
                serde_json::json!({"type":"ProportionalRoPEConfig","base":1000000.0,
                    "head_dim":512,"rotary_dim":128,"max_sequence_length":131072})
            } else {
                serde_json::json!({"type":"UnscaledRoPEConfig","base":10000.0,
                "head_dim":256,"max_sequence_length":131072})
            })?;
            let attention: AttentionConfig = serde_json::from_value(serde_json::json!({
                "type":"AttentionConfig","qkvg_projection_config":{},"out_projection_config":{},"query_norm_config":norm_config,
                "key_norm_config":null,"num_heads":4,"num_groups":profile.kv_heads(),"head_dim":width,
                // Source flips an inclusive-distance<=512 bidirectional mask:
                // one query may see 513 supplied keys. Ordinary target rings
                // retain their own 512-row policy; this component owns no KV.
                "is_causal":true,"scale":1.0,"sliding_window_size":if global {None} else {Some(513)},
                "logit_soft_cap":null,"has_sinks":false,"has_qkvg_biases":false,"has_out_biases":false,
                "has_gate":false,"normalize_values":true,"is_kv_sharing":true,"materialize_rope_bf16":true
            }))?;
            let eager = ReferenceBF16Attention::for_committed_target(context, &attention)
                .map_err(Gemma4AssistantError::Backend)?;
            let (attention, factors) = Attention::new(
                DIM,
                DataType::BF16,
                Some(&rope),
                &attention,
                &tree.subtree(&base).subtree("mixer"),
                context,
            )?;
            if factors.is_some() {
                return Err(Gemma4AssistantError::Invalid);
            }
            let mlp: Box<dyn Mlp<B>> = Box::new(AssistantSourceGeluMlp {
                gelu: <B::Kernels as Kernels>::VisionGeluBF16Kernel::new(context)
                    .map_err(Gemma4AssistantError::Backend)?,
                multiply: GatedActMul::<B>::full_precision(
                    context,
                    DataType::BF16,
                    false,
                    false,
                    GatedActMulSettings::default(),
                )
                .map_err(Gemma4AssistantError::Backend)?,
                reference: ReferenceBF16Matmul::new(context).map_err(Gemma4AssistantError::Backend)?,
                up_weights: tree
                    .subtree(&base)
                    .subtree("mlp.up_projection.weights")
                    .leaf("weights")?
                    .validate(&[4096, DIM], DataType::BF16)?
                    .read_shared_weight()?,
                down_weights: tree
                    .subtree(&base)
                    .subtree("mlp.down_projection.weights")
                    .leaf("weights")?
                    .validate(&[DIM, 2048], DataType::BF16)?
                    .read_shared_weight()?,
            });
            let scalar =
                tree.subtree(&base).leaf("post_layer_scalar")?.validate(&[1], DataType::BF16)?.read_slice::<bf16>()?[0]
                    .to_f32();
            if !scalar.is_finite() {
                return Err(Gemma4AssistantError::Invalid);
            }
            layers.push(AssistantLayer {
                pre_attention: norm(&format!("{base}.pre_mixer_norm"), DIM)?,
                attention,
                eager,
                post_attention: norm(&format!("{base}.post_mixer_norm"), DIM)?,
                pre_mlp: norm(&format!("{base}.pre_mlp_norm"), DIM)?,
                mlp,
                post_mlp: norm(&format!("{base}.post_mlp_norm"), DIM)?,
                scalar,
                rope,
            });
        }
        let raw = tree.leaf("token_ordering")?.validate(&[VOCAB], DataType::I64)?.read_slice::<i64>()?;
        let mut seen = vec![false; VOCAB as usize];
        let ordering = raw
            .into_iter()
            .map(|index| {
                let index = u32::try_from(index).ok().filter(|&n| n < VOCAB).ok_or(Gemma4AssistantError::Invalid)?;
                if std::mem::replace(&mut seen[index as usize], true) {
                    return Err(Gemma4AssistantError::Invalid);
                }
                Ok(index)
            })
            .collect::<Result<Vec<_>, Gemma4AssistantError<B>>>()?
            .into_boxed_slice();
        Ok(Self {
            profile,
            target_identity: None,
            pre_projection: linear("pre_projection", 2 * profile.backbone_dim(), DIM)?,
            post_projection: linear("post_projection", DIM, profile.backbone_dim())?,
            layers: layers.into_boxed_slice(),
            output_norm: norm("output_norm", DIM)?,
            centroids: linear("centroids", DIM, CENTROIDS)?,
            readout: linear("readout", DIM, VOCAB)?,
            ordering,
            add: <B::Kernels as Kernels>::MaterializedBF16AddScaleKernel::new(context, false)
                .map_err(Gemma4AssistantError::Backend)?,
        })
    }

    /// Reference component uses the full device readout before source centroid
    /// masking. It establishes no sparse-readout or end-to-end speed claim.
    pub fn encode_step(
        &self,
        embedding: impl BufferRef<Backend = B>,
        hidden: impl BufferRef<Backend = B>,
        local: &CommittedAttentionKV<'_, B>,
        global: &CommittedAttentionKV<'_, B>,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Gemma4AssistantStep<B>, Gemma4AssistantError<B>> {
        let row_bytes = self.profile.backbone_dim() as usize * 2;
        if embedding.size() != row_bytes
            || hidden.size() != row_bytes
            || local.position() != global.position()
            || local.owner_layer() == global.owner_layer()
        {
            return Err(Gemma4AssistantError::Invalid);
        }
        let backend = Gemma4AssistantError::Backend;
        let mut joined = commands.allocate_scratch(2 * row_bytes).map_err(backend)?;
        commands.encode_copy(embedding, joined.subrange_mut(..row_bytes));
        commands.encode_copy(hidden, joined.subrange_mut(row_bytes..));
        let mut value = self.pre_projection.encode(joined, 1, commands).map_err(backend)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let target = if index == 3 {
                global
            } else {
                local
            };
            let rope =
                PrecalculatedRoPE::<B>::precalculate(&layer.rope, &[target.position()], commands).map_err(backend)?;
            let norm =
                layer.pre_attention.encode(&value, 0, 1, None::<&mut B::ScratchBuffer>, commands).map_err(backend)?;
            let attention = layer
                .attention
                .attend_committed_target_eager(norm, Some(&rope), target, &layer.eager, commands)
                .map_err(backend)?;
            let attention = layer
                .post_attention
                .encode(&attention, 0, 1, None::<&mut B::ScratchBuffer>, commands)
                .map_err(backend)?;
            let mut residual = commands.allocate_scratch(DIM as usize * 2).map_err(backend)?;
            self.add.encode(Some(&attention), &value, &mut residual, DIM, DIM, 1.0, commands);
            let mlp =
                layer.pre_mlp.encode(&residual, 0, 1, None::<&mut B::ScratchBuffer>, commands).map_err(backend)?;
            let mlp = layer.mlp.encode(mlp, 1, commands).map_err(backend)?;
            let mlp = layer.post_mlp.encode(&mlp, 0, 1, None::<&mut B::ScratchBuffer>, commands).map_err(backend)?;
            value = commands.allocate_scratch(DIM as usize * 2).map_err(backend)?;
            self.add.encode(Some(&mlp), &residual, &mut value, DIM, DIM, layer.scalar, commands);
        }
        let value = self.output_norm.encode(&value, 0, 1, None::<&mut B::ScratchBuffer>, commands).map_err(backend)?;
        let mut copy = commands.allocate_scratch(DIM as usize * 2).map_err(backend)?;
        commands.encode_copy(&value, &mut copy);
        let feedback = self.post_projection.encode(copy, 1, commands).map_err(backend)?;
        let mut copy = commands.allocate_scratch(DIM as usize * 2).map_err(backend)?;
        commands.encode_copy(&value, &mut copy);
        let centroids = self.centroids.encode(copy, 1, commands).map_err(backend)?;
        let logits = self.readout.encode(value, 1, commands).map_err(backend)?;
        Ok(Gemma4AssistantStep {
            feedback,
            centroids,
            logits,
        })
    }

    /// Apply source top-centroid vocabulary masking to completed BF16 readbacks.
    pub fn mask_logits(
        &self,
        centroids: &[bf16],
        logits: &[bf16],
    ) -> Result<Vec<f32>, Gemma4AssistantError<B>> {
        if centroids.len() != CENTROIDS as usize
            || logits.len() != VOCAB as usize
            || centroids.iter().chain(logits).any(|v| !v.is_finite())
        {
            return Err(Gemma4AssistantError::Invalid);
        }
        let mut indices = (0..CENTROIDS as usize).collect::<Vec<_>>();
        indices.sort_unstable_by(|&a, &b| centroids[b].to_f32().total_cmp(&centroids[a].to_f32()).then(a.cmp(&b)));
        let selected = indices[..TOP_K]
            .iter()
            .flat_map(|&cluster| self.ordering[cluster * 128..(cluster + 1) * 128].iter().copied())
            .collect::<Vec<_>>();
        let minimum = selected.iter().map(|&index| logits[index as usize].to_f32()).fold(f32::INFINITY, f32::min);
        let mut output = vec![bf16::from_f32(minimum - 1.0).to_f32(); VOCAB as usize];
        for index in selected {
            output[index as usize] = logits[index as usize].to_f32();
        }
        Ok(output)
    }

    /// Diagnostic duplicate of the ordinary step, with ephemeral stage copies.
    /// Tests must require the returned final planes to match ordinary encoding.
    #[cfg(test)]
    pub(crate) fn encode_step_traced(
        &self,
        embedding: impl BufferRef<Backend = B>,
        hidden: impl BufferRef<Backend = B>,
        local: &CommittedAttentionKV<'_, B>,
        global: &CommittedAttentionKV<'_, B>,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(Gemma4AssistantStep<B>, Vec<(String, B::ScratchBuffer)>), Gemma4AssistantError<B>> {
        let backend = Gemma4AssistantError::Backend;
        let mut stages = Vec::new();
        let row_bytes = self.profile.backbone_dim() as usize * 2;
        if embedding.size() != row_bytes || hidden.size() != row_bytes || local.position() != global.position() {
            return Err(Gemma4AssistantError::Invalid);
        }
        let mut joined = commands.allocate_scratch(2 * row_bytes).map_err(backend)?;
        commands.encode_copy(embedding, joined.subrange_mut(..row_bytes));
        commands.encode_copy(hidden, joined.subrange_mut(row_bytes..));
        let mut value = self.pre_projection.encode(joined, 1, commands).map_err(backend)?;
        trace_stage::<B>("pre_projection".into(), &value, &mut stages, commands).map_err(backend)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let target = if index == 3 {
                global
            } else {
                local
            };
            let rope =
                PrecalculatedRoPE::<B>::precalculate(&layer.rope, &[target.position()], commands).map_err(backend)?;
            let norm =
                layer.pre_attention.encode(&value, 0, 1, None::<&mut B::ScratchBuffer>, commands).map_err(backend)?;
            trace_stage::<B>(format!("layer{index}.pre_attention"), &norm, &mut stages, commands).map_err(backend)?;
            let mut query_input = commands.allocate_scratch(norm.size()).map_err(backend)?;
            commands.encode_copy(&norm, &mut query_input);
            let query_stages = layer
                .attention
                .assistant_query_trace(query_input, &rope, target, &layer.eager, commands)
                .map_err(backend)?;
            stages.extend(query_stages.into_iter().map(|(name, buffer)| (format!("layer{index}.{name}"), buffer)));
            let attention = layer
                .attention
                .attend_committed_target_eager(norm, Some(&rope), target, &layer.eager, commands)
                .map_err(backend)?;
            trace_stage::<B>(format!("layer{index}.attention_out"), &attention, &mut stages, commands)
                .map_err(backend)?;
            let attention = layer
                .post_attention
                .encode(&attention, 0, 1, None::<&mut B::ScratchBuffer>, commands)
                .map_err(backend)?;
            trace_stage::<B>(format!("layer{index}.post_attention"), &attention, &mut stages, commands)
                .map_err(backend)?;
            let mut residual = commands.allocate_scratch(DIM as usize * 2).map_err(backend)?;
            self.add.encode(Some(&attention), &value, &mut residual, DIM, DIM, 1.0, commands);
            trace_stage::<B>(format!("layer{index}.residual"), &residual, &mut stages, commands).map_err(backend)?;
            let mlp =
                layer.pre_mlp.encode(&residual, 0, 1, None::<&mut B::ScratchBuffer>, commands).map_err(backend)?;
            trace_stage::<B>(format!("layer{index}.pre_mlp"), &mlp, &mut stages, commands).map_err(backend)?;
            let mut mlp_trace_input = commands.allocate_scratch(mlp.size()).map_err(backend)?;
            commands.encode_copy(&mlp, &mut mlp_trace_input);
            let mlp_stages = layer
                .mlp
                .trace_dense(mlp_trace_input, 1, commands)
                .map_err(backend)?
                .ok_or(Gemma4AssistantError::Invalid)?;
            stages.extend(mlp_stages.into_iter().map(|(name, buffer)| (format!("layer{index}.{name}"), buffer)));
            let mlp = layer.mlp.encode(mlp, 1, commands).map_err(backend)?;
            trace_stage::<B>(format!("layer{index}.mlp"), &mlp, &mut stages, commands).map_err(backend)?;
            let mlp = layer.post_mlp.encode(&mlp, 0, 1, None::<&mut B::ScratchBuffer>, commands).map_err(backend)?;
            trace_stage::<B>(format!("layer{index}.post_mlp"), &mlp, &mut stages, commands).map_err(backend)?;
            value = commands.allocate_scratch(DIM as usize * 2).map_err(backend)?;
            self.add.encode(Some(&mlp), &residual, &mut value, DIM, DIM, layer.scalar, commands);
            trace_stage::<B>(format!("layer{index}.output"), &value, &mut stages, commands).map_err(backend)?;
        }
        let value = self.output_norm.encode(&value, 0, 1, None::<&mut B::ScratchBuffer>, commands).map_err(backend)?;
        trace_stage::<B>("output_norm".into(), &value, &mut stages, commands).map_err(backend)?;
        let mut copy = commands.allocate_scratch(DIM as usize * 2).map_err(backend)?;
        commands.encode_copy(&value, &mut copy);
        let feedback = self.post_projection.encode(copy, 1, commands).map_err(backend)?;
        let mut copy = commands.allocate_scratch(DIM as usize * 2).map_err(backend)?;
        commands.encode_copy(&value, &mut copy);
        let centroids = self.centroids.encode(copy, 1, commands).map_err(backend)?;
        let logits = self.readout.encode(value, 1, commands).map_err(backend)?;
        Ok((
            Gemma4AssistantStep {
                feedback,
                centroids,
                logits,
            },
            stages,
        ))
    }
}

#[cfg(test)]
fn trace_stage<B: Backend>(
    name: String,
    value: &B::ScratchBuffer,
    stages: &mut Vec<(String, B::ScratchBuffer)>,
    commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
) -> Result<(), B::Error> {
    let mut copy = commands.allocate_scratch(value.size())?;
    commands.encode_copy(value, &mut copy);
    stages.push((name, copy));
    Ok(())
}
