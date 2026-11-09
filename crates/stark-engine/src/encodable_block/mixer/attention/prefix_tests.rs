//! Synthetic payloads exercise physical ring overwrite and traversal restoration.
use stark_engine_macros::stark_test;

use super::*;
use crate::backends::common::{CommandBufferExecutable, CommandBufferPending};

fn state<B: Backend>(
    context: &B::Context,
    capacity: u32,
) -> AttentionState<B> {
    let rows = capacity + ATTENTION_SUFFIX_CAPACITY;
    AttentionState {
        element_dim: 4,
        data_type: DataType::F32,
        cache: KVCacheState::ring(capacity),
        capacity_elements: rows,
        is_sparse: false,
        keys: Box::new(context.create_buffer(rows as usize * 16).unwrap()),
        values: Box::new(context.create_buffer(rows as usize * 16).unwrap()),
        kv_cache_update: <B::Kernels as Kernels>::KVCacheUpdateKernel::new(context, DataType::F32).unwrap(),
    }
}

fn buffer_mut<B: Backend>(buffer: &mut Box<dyn Buffer<Backend = B>>) -> &mut B::GlobalBuffer {
    (buffer.as_mut() as &mut dyn Any).downcast_mut::<B::GlobalBuffer>().unwrap()
}

fn round_trip<B: Backend>(wrapped: bool) {
    let context = B::Context::new().unwrap();
    let mut live = state::<B>(&context, 8);
    let accepted = if wrapped {
        13
    } else {
        3
    };
    live.cache.accept(&(0..accepted).collect::<Vec<_>>());
    // Single updates alter the offset while preserving the last bulk origin.
    if wrapped {
        live.cache.accept(&[0]);
        live.cache.accept(&[0]);
    }
    let view = live.view();
    let bytes = view.prefix_len() as usize * 16;
    let key_values: Vec<f32> = (0..bytes / 4).map(|i| i as f32 * 0.25 + 1.0).collect();
    let value_values: Vec<f32> = (0..bytes / 4).map(|i| -(i as f32) * 0.5 - 1.0).collect();
    buffer_mut(&mut live.keys).subrange_mut(..bytes).copyin(&key_values);
    buffer_mut(&mut live.values).subrange_mut(..bytes).copyin(&value_values);
    let mut command = context.create_command_buffer(None, None).unwrap();
    let saved = live
        .capture_prefix(
            accepted
                + if wrapped {
                    2
                } else {
                    0
                },
            &context,
            &mut command,
        )
        .unwrap()
        .unwrap();
    command.end_encoding().submit().wait_until_completed().unwrap();
    assert_eq!(saved.payload_bytes(), bytes * 2);
    for _ in 0..2 {
        // Overwrite every retained physical row and all traversal scalars.
        buffer_mut(&mut live.keys).subrange_mut(..bytes).copyin(&vec![777.0f32; bytes / 4]);
        buffer_mut(&mut live.values).subrange_mut(..bytes).copyin(&vec![888.0f32; bytes / 4]);
        live.cache.accept(&(0..17).collect::<Vec<_>>());
        assert!(saved.can_restore(&live));
        let mut command = context.create_command_buffer(None, None).unwrap();
        assert!(saved.restore(&mut live, &mut command));
        command.end_encoding().submit().wait_until_completed().unwrap();
        assert_eq!(live.view().prefix_len(), view.prefix_len());
        assert_eq!(live.view().ring_params().unwrap().ring_offset, view.ring_params().unwrap().ring_offset);
        assert_eq!(live.view().source_ring_origin(), view.source_ring_origin());
        let keys = (live.keys.as_ref() as &dyn Any).downcast_ref::<B::GlobalBuffer>().unwrap();
        let values = (live.values.as_ref() as &dyn Any).downcast_ref::<B::GlobalBuffer>().unwrap();
        assert!((&keys).subrange(..bytes).as_slice::<f32>() == key_values, "ring key payload differs");
        assert!((&values).subrange(..bytes).as_slice::<f32>() == value_values, "ring value payload differs");
    }
    let mut incompatible = state::<B>(&context, 9);
    assert!(!saved.can_restore(&incompatible));
    let mut command = context.create_command_buffer(None, None).unwrap();
    assert!(!saved.restore(&mut incompatible, &mut command));
    live.element_dim += 1;
    assert!(!saved.can_restore(&live));
    live.element_dim -= 1;
    live.data_type = DataType::BF16;
    assert!(!saved.can_restore(&live));
    live.data_type = DataType::F32;
    live.cache = KVCacheState::full();
    assert!(!saved.can_restore(&live));
}

#[cfg(backend = "cpu")]
#[stark_test]
fn ring_prefix_owned_payload_cpu() {
    round_trip::<crate::backends::cpu::Cpu>(false);
    round_trip::<crate::backends::cpu::Cpu>(true);
}

#[cfg(backend = "metal")]
#[stark_test]
fn ring_prefix_owned_payload_metal() {
    round_trip::<crate::backends::metal::Metal>(false);
    round_trip::<crate::backends::metal::Metal>(true);
}

fn committed_external_view<B: Backend>() {
    use crate::{
        backends::common::kernel::{AttentionArguments, AttentionKernel, AttentionKernelConfig},
        encodable_block::mixer::attention::CommittedAttentionKV,
    };
    let context = B::Context::new().unwrap();
    for ring in [false, true] {
        let rows = 4 + ATTENTION_SUFFIX_CAPACITY;
        let mut live = AttentionState::<B> {
            element_dim: 64,
            data_type: DataType::F32,
            cache: if ring {
                KVCacheState::ring(4)
            } else {
                KVCacheState::full()
            },
            capacity_elements: rows,
            is_sparse: false,
            keys: Box::new(context.create_buffer(rows as usize * 256).unwrap()),
            values: Box::new(context.create_buffer(rows as usize * 256).unwrap()),
            kv_cache_update: <B::Kernels as Kernels>::KVCacheUpdateKernel::new(&context, DataType::F32).unwrap(),
        };
        let position = if ring {
            15
        } else {
            3
        };
        if ring {
            live.cache.accept(&(0..13).collect::<Vec<_>>());
            live.cache.accept(&[0]);
            live.cache.accept(&[0]);
        } else {
            live.cache.accept(&[0, 1, 2]);
        }
        let retained = live.view().prefix_len() as usize;
        buffer_mut(&mut live.keys).copyin(&vec![0.0f32; rows as usize * 64]);
        // A phantom suffix read would ingest 1000 instead of the final real
        // committed row. Distinct rows also expose incorrect ring traversal.
        buffer_mut(&mut live.values).copyin(&vec![1000.0f32; rows as usize * 64]);
        let actual: Vec<f32> = (0..retained).flat_map(|i| std::iter::repeat_n((i + 1) as f32, 64)).collect();
        buffer_mut(&mut live.values).subrange_mut(..retained * 256).copyin(&actual);
        let view = CommittedAttentionKV::new(&live, 7, position).unwrap();
        assert_eq!(view.owner_layer(), 7);
        assert_eq!(view.position(), position);
        assert_eq!(view.retained_rows(), retained as u32);
        assert_eq!(view.query_cache_view().prefix_len(), retained as u32 - 1);
        assert_eq!(
            view.scratch_bytes(),
            Some(if ring {
                retained * 512
            } else {
                0
            })
        );
        assert!(CommittedAttentionKV::new(&live, 7, 0).is_none());
        if !ring {
            assert!(CommittedAttentionKV::new(&live, 7, position + 1).is_none());
        }
        let mut command = context.create_command_buffer(None, None).unwrap();
        let mut queries = context.create_buffer(256).unwrap();
        queries.copyin(&[0.0f32; 64]);
        let kernel = <B::Kernels as Kernels>::AttentionKernel::new(
            &context,
            AttentionKernelConfig {
                head_dim: 64,
                num_groups: 1,
                num_q_heads: 1,
                has_sinks: false,
                is_kv_cache_ring: ring,
                is_causal: true,
                sliding_window_size: ring.then_some(4),
                scale: Some(1.0),
                data_type: DataType::F32,
                materialize_prefill_scores: false,
                source_bf16_prefill: false,
            },
        )
        .unwrap();
        // Scratch owners must remain live through the submission using them.
        // A second command must not borrow scratch from a dropped first pool.
        let temporal = if ring {
            Some(view.linearize_ring(&mut command).unwrap())
        } else {
            None
        };
        let mut values_readback = if ring {
            Some(context.create_buffer(retained * 256).unwrap())
        } else {
            None
        };
        let output = if let Some((keys, values)) = &temporal {
            command.encode_copy(values, values_readback.as_mut().unwrap());
            kernel
                .encode(
                    AttentionArguments {
                        queries: &queries,
                        keys,
                        values,
                        suffix_length: 1,
                        trie: None::<&B::ConstantBuffer>,
                        sinks: None,
                        cache: view.query_cache_view(),
                        image_span: None,
                    },
                    &mut command,
                )
                .unwrap()
        } else {
            kernel
                .encode(
                    AttentionArguments {
                        queries: &queries,
                        keys: live.keys.as_ref(),
                        values: live.values.as_ref(),
                        suffix_length: 1,
                        trie: None::<&B::ConstantBuffer>,
                        sinks: None,
                        cache: view.query_cache_view(),
                        image_span: None,
                    },
                    &mut command,
                )
                .unwrap()
        };
        let mut result = context.create_buffer(256).unwrap();
        command.encode_copy(&output, &mut result);
        command.end_encoding().submit().wait_until_completed().unwrap();
        if let Some(values_readback) = values_readback {
            let offset = view.view.ring_params().unwrap().ring_offset as usize;
            let expected: Vec<f32> =
                (0..retained).flat_map(|i| std::iter::repeat_n(((i + offset) % retained + 1) as f32, 64)).collect();
            assert!(values_readback.as_slice::<f32>() == expected);
        }
        let expected_mean = (retained + 1) as f32 / 2.0;
        let observed = result.as_slice::<f32>();
        let error = observed.iter().map(|&v| (v - expected_mean).abs()).fold(0.0f32, f32::max);
        assert!(
            observed.iter().all(|v| v.is_finite()) && error < 1e-5,
            "committed query ring={ring}, expected={expected_mean}, max_error={error}, first={}",
            observed[0]
        );
        assert_eq!(live.view().prefix_len(), retained as u32);
    }
}

#[cfg(backend = "cpu")]
#[stark_test]
fn committed_external_kv_excludes_phantom_suffix_cpu() {
    committed_external_view::<crate::backends::cpu::Cpu>();
}

#[cfg(backend = "metal")]
#[stark_test]
fn committed_external_kv_excludes_phantom_suffix_metal() {
    committed_external_view::<crate::backends::metal::Metal>();
}

fn committed_bf16_eager_view<B: Backend>() {
    use half::bf16;

    use crate::{
        config::token_mixer::attention::AttentionConfig,
        encodable_block::mixer::attention::{CommittedAttentionKV, reference::ReferenceBF16Attention},
    };
    let context = B::Context::new().unwrap();
    for ring in [false, true] {
        let capacity = 4 + ATTENTION_SUFFIX_CAPACITY;
        let mut live = AttentionState::<B> {
            element_dim: 64,
            data_type: DataType::BF16,
            cache: if ring {
                KVCacheState::ring(4)
            } else {
                KVCacheState::full()
            },
            capacity_elements: capacity,
            is_sparse: false,
            keys: Box::new(context.create_buffer(capacity as usize * 128).unwrap()),
            values: Box::new(context.create_buffer(capacity as usize * 128).unwrap()),
            kv_cache_update: <B::Kernels as Kernels>::KVCacheUpdateKernel::new(&context, DataType::BF16).unwrap(),
        };
        if ring {
            live.cache.accept(&(0..13).collect::<Vec<_>>());
            live.cache.accept(&[0]);
            live.cache.accept(&[0]);
        } else {
            live.cache.accept(&[0, 1, 2]);
        }
        let count = live.view().prefix_len();
        buffer_mut(&mut live.keys).copyin(&vec![bf16::ZERO; capacity as usize * 64]);
        buffer_mut(&mut live.values).copyin(&vec![bf16::from_f32(1000.0); capacity as usize * 64]);
        let actual =
            (0..count).flat_map(|i| std::iter::repeat_n(bf16::from_f32((i + 1) as f32), 64)).collect::<Vec<_>>();
        buffer_mut(&mut live.values).subrange_mut(..count as usize * 128).copyin(&actual);
        let view = CommittedAttentionKV::new(
            &live,
            7,
            if ring {
                15
            } else {
                3
            },
        )
        .unwrap();
        let config: AttentionConfig = serde_json::from_value(serde_json::json!({
            "type":"AttentionConfig","qkvg_projection_config":{},"out_projection_config":{},
            "query_norm_config":null,"key_norm_config":null,"num_heads":1,"num_groups":1,"head_dim":64,
            "is_causal":true,"scale":1.0,"sliding_window_size":if ring {Some(513)} else {None},
            "logit_soft_cap":null,"has_sinks":false,"has_qkvg_biases":false,"has_out_biases":false,
            "has_gate":false,"normalize_values":false,"is_kv_sharing":true}))
        .unwrap();
        // Ordinary reference admission must continue to reject shared/ring state.
        assert!(ReferenceBF16Attention::<B>::new(&context, &config).is_err());
        let eager = ReferenceBF16Attention::<B>::for_committed_target(&context, &config).unwrap();
        assert!(eager.matches_geometry(1, 1, 64));
        assert!(!eager.matches_geometry(2, 1, 64));
        let mut command = context.create_command_buffer(None, None).unwrap();
        let mut query = context.create_buffer(128).unwrap();
        query.copyin(&[bf16::ZERO; 64]);
        let (keys, values) = view.validation_copy(&mut command).unwrap();
        let output = eager.encode(&query, &keys, &values, 1, count - 1, &mut command).unwrap();
        let mut result = context.create_buffer(128).unwrap();
        command.encode_copy(&output, &mut result);
        command.end_encoding().submit().wait_until_completed().unwrap();
        let probability = bf16::from_f32(1.0 / count as f32).to_f32();
        let expected = bf16::from_f32(probability * (count * (count + 1) / 2) as f32);
        assert!(result.as_slice::<bf16>().iter().all(|&v| v == expected));
        assert_eq!(live.view().prefix_len(), count);
    }
}

#[cfg(backend = "cpu")]
#[stark_test]
fn committed_bf16_eager_excludes_phantom_suffix_cpu() {
    committed_bf16_eager_view::<crate::backends::cpu::Cpu>();
}
#[cfg(backend = "metal")]
#[stark_test]
fn committed_bf16_eager_excludes_phantom_suffix_metal() {
    committed_bf16_eager_view::<crate::backends::metal::Metal>();
}
