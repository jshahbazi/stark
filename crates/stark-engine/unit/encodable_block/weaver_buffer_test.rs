use stark_engine_macros::stark_test;

use crate::{
    backends::common::{
        Backend, Buffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
    },
    tests::helpers::{buffer_to_vec, create_context, for_each_backend},
};

fn initialized_rows<B: Backend>() {
    let context = create_context::<B>();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    let mut old = command_buffer.allocate_scratch(24).unwrap();
    command_buffer.encode_fill(&mut old, 0xff);
    drop(old);
    let (ids, logits) = super::allocate_node_candidates::<B>(&mut command_buffer, 2, 3).unwrap();
    let mut ids_readback = context.create_buffer(ids.size()).unwrap();
    let mut logits_readback = context.create_buffer(logits.size()).unwrap();
    command_buffer.encode_copy(&ids, &mut ids_readback);
    command_buffer.encode_copy(&logits, &mut logits_readback);
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    assert_eq!(buffer_to_vec::<B, u32>(&ids_readback), vec![0; 6]);
    assert_eq!(buffer_to_vec::<B, f32>(&logits_readback), vec![0.0; 6]);
}

#[stark_test]
fn unexpanded_weaver_rows_have_initialized_gather_ids_and_logits() {
    for_each_backend!(|B| initialized_rows::<B>());
}
