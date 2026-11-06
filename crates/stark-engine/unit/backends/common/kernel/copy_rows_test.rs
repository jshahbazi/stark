use stark_engine_macros::stark_test;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        kernel::CopyRowsKernel,
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, for_each_backend},
};

#[stark_test]
fn test_copy_rows_preserves_disjoint_columns_and_raw_bits() {
    for_each_backend!(|B| {
        let context = <B as Backend>::Context::new().unwrap();
        let kernel = <<B as Backend>::Kernels as Kernels>::CopyRowsKernel::new(&context).unwrap();
        let input: Vec<u8> = (0..35).map(|x| (x * 73) as u8).collect();
        let source = create_buffer_with_data::<B, u8>(&context, &input);
        let mut target = create_buffer_with_data::<B, u8>(&context, &[0xa5; 65]);
        let mut command = context.create_command_buffer(None, None).unwrap();
        kernel.encode(&source, &mut target, 7, 13, 3, 35, &mut command);
        command.end_encoding().submit().wait_until_completed().unwrap();
        let actual = buffer_to_vec::<B, u8>(&target);
        let mut expected = vec![0xa5; 65];
        for row in 0..5 {
            expected[row * 13 + 3..row * 13 + 10].copy_from_slice(&input[row * 7..row * 7 + 7]);
        }
        assert_eq!(actual, expected);
    });
}
