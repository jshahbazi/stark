#include <metal_stdlib>
#include "common/dsl.h"

// Copy disjoint contiguous row segments without an encoder/blit per row.
PUBLIC KERNEL(CopyRows)(
    const device uchar* input,
    device uchar* output,
    constant uint& row_bytes,
    constant uint& output_row_bytes,
    constant uint& output_offset_bytes,
    constant uint& length,
    const uint index AXIS(length, 128)
) {
  const uint destination = (index / row_bytes) * output_row_bytes + output_offset_bytes + index % row_bytes;
  output[destination] = input[index];
}
