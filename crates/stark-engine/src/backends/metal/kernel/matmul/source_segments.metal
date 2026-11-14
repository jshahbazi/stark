// The BF16 partition reduction follows MLX v0.32.3
// mlx/backend/metal/kernels/reduction/reduce_col.h (Apple Inc., MIT).
// Retain its BF16 running totals and the same 32-lane reduction inputs.
// The intrinsic is qualified on the physical device; its internal arithmetic
// is not inferred from the source-level BF16 argument type.
#include "../common/dsl.h"

using namespace metal;

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

KERNEL(SourceBF16SplitReduce)(
    const device bfloat* partials,
    device bfloat* output,
    constant uint& element_count,
    constant uint& partition_count,
    const uint index AXIS(element_count * 32, 256)
) {
    const uint column = index / 32;
    const uint lane = index % 32;
    // MLX selects col_reduce_small before the SIMD path for fewer than
    // 32 partitions. It uses up to eight strided BF16 running totals,
    // then combines those totals in lane order with BF16 rounding.
    if (partition_count < 32) {
        if (lane == 0) {
            bfloat totals[8];
            const uint height = min(partition_count, 8u);
            for (uint row = 0; row < height; ++row) {
                totals[row] = bfloat(0.0f);
                for (uint partition = row; partition < partition_count; partition += height) {
                    totals[row] = bfloat(float(partials[partition * element_count + column]) + float(totals[row]));
                }
            }
            bfloat result = totals[0];
            for (uint row = 1; row < height; ++row) {
                result = bfloat(float(totals[row]) + float(result));
            }
            output[column] = result;
        }
        return;
    }
    bfloat total = bfloat(0.0f);
    for (uint partition = lane; partition < partition_count; partition += 32) {
        total = bfloat(float(total) + float(partials[partition * element_count + column]));
    }
    // MLX bf16_math.h instantiates simd_sum(bfloat16_t) by promoting
    // its input to float and casting the SIMD result back to BF16.
    const bfloat result = bfloat(simd_sum(float(total)));
    if (lane == 0) {
        output[column] = result;
    }
}

KERNEL(SourceScatterRows)(
    const device bfloat* input,
    device bfloat* output,
    constant uint& m,
    constant uint& segment_width,
    constant uint& output_width,
    constant uint& column_offset,
    const uint index AXIS(m * segment_width, 256)
) {
    output[(index / segment_width) * output_width + column_offset + index % segment_width] = input[index];
}
