use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::common::{BufferMut, BufferRef, Kernels, kernel::TensorAddScaleKernel},
    data_type::DataType,
};

#[stark_test]
fn argument_tables_are_exclusive_until_completion() {
    let context = MetalContext::new().unwrap();
    let first = context.create_command_buffer(None, None).unwrap();
    let first_table = first.argument_table.clone();
    let second = context.create_command_buffer(None, None).unwrap();
    let second_table = second.argument_table.clone();
    assert!(!std::ptr::eq(&*first_table, &*second_table));
    assert!(context.command_buffer_cache.lock().is_empty());

    let first_pending = first.end_encoding().submit();
    let second_pending = second.end_encoding().submit();
    first_pending.wait_until_completed().unwrap();
    second_pending.wait_until_completed().unwrap();
    assert_eq!(context.command_buffer_cache.lock().len(), 2);

    let recycled = context.create_command_buffer(None, None).unwrap();
    assert!(
        std::ptr::eq(&*recycled.argument_table, &*first_table)
            || std::ptr::eq(&*recycled.argument_table, &*second_table)
    );
    recycled.end_encoding().submit().wait_until_completed().unwrap();
}

#[stark_test]
fn recycled_argument_tables_rebind_buffers_constants_and_optional_inputs() {
    let context = MetalContext::new().unwrap();
    let out_of_place =
        <<Metal as Backend>::Kernels as Kernels>::TensorAddScaleKernel::new(&context, DataType::F32, false).unwrap();
    let in_place =
        <<Metal as Backend>::Kernels as Kernels>::TensorAddScaleKernel::new(&context, DataType::F32, true).unwrap();
    // Retain earlier buffers so stale bindings cannot accidentally refer to a
    // new allocation at the same address.
    let mut retained = Vec::new();
    for iteration in 0..8 {
        let values = [iteration as f32 + 1.0; 8];
        let biases = [2.0f32; 8];
        let mut input = context.create_buffer(32).unwrap();
        input.copyin(&values);
        let mut bias = context.create_buffer(32).unwrap();
        bias.copyin(&biases);
        let mut output = context.create_buffer(32).unwrap();
        output.copyin(&values);
        let scale = iteration as f32 + 0.5;
        let mut command = context.create_command_buffer(None, None).unwrap();
        // Vary constant offsets even when the global allocator reuses a page.
        drop(command.allocate_constant((iteration + 1) * 64).unwrap());
        if iteration % 2 == 0 {
            out_of_place.encode(Some(&input), &bias, &mut output, 8, 8, scale, &mut command);
        } else {
            in_place.encode(None::<&<Metal as Backend>::GlobalBuffer>, &bias, &mut output, 8, 8, scale, &mut command);
        }
        command.end_encoding().submit().wait_until_completed().unwrap();
        assert_eq!(output.copyout::<f32>(), vec![(values[0] + biases[0]) * scale; 8]);
        retained.push((input, bias, output));
    }
}
