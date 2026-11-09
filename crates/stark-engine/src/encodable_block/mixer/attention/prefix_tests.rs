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
