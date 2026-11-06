use stark_engine_macros::kernel;

/// Bit-preserving copy into a disjoint output-column interval.
#[kernel(CopyRows)]
pub fn copy_rows(
    input: *const u8,
    output: *mut u8,
    row_bytes: u32,
    output_row_bytes: u32,
    output_offset_bytes: u32,
    length: u32,
) {
    for index in 0..length as usize {
        let destination = index / row_bytes as usize * output_row_bytes as usize
            + output_offset_bytes as usize
            + index % row_bytes as usize;
        unsafe {
            *output.add(destination) = *input.add(index);
        }
    }
}
