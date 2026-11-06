//! Snapshot copy/restore checks use synthetic state, never evaluation payloads.
use stark_engine_macros::stark_test;

use super::*;
use crate::backends::common::{BufferMut, BufferRef, CommandBufferExecutable, CommandBufferPending};

fn round_trip<B: Backend>() {
    let context = B::Context::new().unwrap();
    let conv: Vec<f32> = (0..1152).map(|x| x as f32 * 0.25 - 1.0).collect();
    let ssm: Vec<f32> = (0..16384).map(|x| -(x as f32) * 0.125).collect();
    let mut conv_state = context.create_buffer(conv.len() * 4).unwrap();
    let mut ssm_state = context.create_buffer(ssm.len() * 4).unwrap();
    conv_state.copyin(&conv);
    ssm_state.copyin(&ssm);
    let mut state = DeltaNetState::<B> {
        conv_state,
        ssm_state,
        suffix_status: None,
        #[cfg(feature = "deferred-state-init")]
        needs_zero_initialization: false,
        state_advance: <B::Kernels as Kernels>::StateAdvanceKernel::new(&context, DataType::BF16, 128, 1, 1).unwrap(),
    };
    let mut command = context.create_command_buffer(None, None).unwrap();
    let snapshot = state.capture_prefix(1024, &context, &mut command).unwrap().unwrap();
    command.end_encoding().submit().wait_until_completed().unwrap();
    assert_eq!(snapshot.payload_bytes(), (conv.len() + ssm.len()) * 4);
    state.conv_state.copyin(&vec![123.0f32; conv.len()]);
    state.ssm_state.copyin(&vec![456.0f32; ssm.len()]);
    state.suffix_status = Some(DeltaNetSuffixStatus::Flat {
        suffix_length: 1,
    });
    assert!(snapshot.can_restore(&state));
    let mut command = context.create_command_buffer(None, None).unwrap();
    assert!(snapshot.restore(&mut state, &mut command));
    command.end_encoding().submit().wait_until_completed().unwrap();
    assert!(state.suffix_status.is_none());
    assert_eq!(state.conv_state.copyout::<f32>(), conv);
    assert_eq!(state.ssm_state.copyout::<f32>(), ssm);
    // Reusing the snapshot after another mutation must still restore its
    // captured value, proving it did not alias the live recurrent buffers.
    state.ssm_state.copyin(&vec![789.0f32; ssm.len()]);
    let mut command = context.create_command_buffer(None, None).unwrap();
    assert!(snapshot.restore(&mut state, &mut command));
    command.end_encoding().submit().wait_until_completed().unwrap();
    assert_eq!(state.ssm_state.copyout::<f32>(), ssm);
    state.suffix_status = Some(DeltaNetSuffixStatus::Flat {
        suffix_length: 1,
    });
    let mut command = context.create_command_buffer(None, None).unwrap();
    assert!(state.capture_prefix(1024, &context, &mut command).unwrap().is_none());
}

#[cfg(backend = "cpu")]
#[stark_test]
fn delta_prefix_snapshot_cpu() {
    round_trip::<crate::backends::cpu::Cpu>();
}

#[cfg(backend = "metal")]
#[stark_test]
fn delta_prefix_snapshot_metal() {
    round_trip::<crate::backends::metal::Metal>();
}
