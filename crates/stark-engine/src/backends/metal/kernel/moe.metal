#include <metal_stdlib>
#include "common/dsl.h"

PUBLIC KERNEL(GemmaMoeRouterScaleBF16)(const device bfloat* input,const device bfloat* scales,
    device bfloat* output,constant uint& width,constant uint& length,constant float& root_scale,
    const uint index AXIS(length,128)) {
    float learned=float(bfloat(float(input[index])*float(scales[index%width])));
    output[index]=bfloat(learned*root_scale);
}

// Stable softmax/top-k. One bounded expert row per thread; only selected
// expert IDs leave this kernel, and every published ID is in range.
PUBLIC KERNEL(MoeRoute)(
    const device float* logits,
    device uint* ids,
    device float* weights,
    constant uint& rows,
    constant uint& experts,
    constant uint& active,
    constant uint& round_weights,
    constant uint& round_logits,
    const uint row AXIS(rows, 1)
) {
    const device float* x = logits + row * experts;
    float maximum = -INFINITY;
    bool valid = experts > 0 && experts <= 1024 && active > 0 && active <= 32 && active <= experts;
    for (uint e = 0; e < experts; ++e) { float value = round_logits ? float(bfloat(x[e])) : x[e]; valid = valid && isfinite(value); maximum = max(maximum, value); }
    float denominator = 0.0f;
    for (uint e = 0; e < experts; ++e) { float value = round_logits ? float(bfloat(x[e])) : x[e]; denominator += exp(value - maximum); }
    float selected_sum = 0.0f;
    for (uint rank = 0; rank < active; ++rank) {
        float best = -INFINITY;
        uint selected = 0;
        for (uint e = 0; e < experts; ++e) {
            bool used = false;
            for (uint previous = 0; previous < rank; ++previous) { used = used || ids[row * active + previous] == e; }
            float value = round_logits ? float(bfloat(x[e])) : x[e];
            if (!used && value > best) { best = value; selected = e; }
        }
        ids[row * active + rank] = valid ? selected : 0;
        float score = exp(best - maximum) / denominator;
        weights[row * active + rank] = score;
        selected_sum += score;
    }
    for (uint rank = 0; rank < active; ++rank) {
        float score = weights[row * active + rank] / selected_sum;
        if (round_weights != 0) { score = float(bfloat(score)); }
        weights[row * active + rank] = valid ? score : NAN;
    }
}

PUBLIC KERNEL(MoeGatherRows)(
    const device uint* ids,
    device uint* indices,
    constant uint& route_offset,
    constant uint& rows_per_expert,
    constant uint& experts,
    constant uint& length,
    const uint position AXIS(length, 32)
) {
    uint expert = ids[route_offset];
    indices[position] = (expert < experts ? expert : 0) * rows_per_expert + position;
}

PUBLIC KERNEL(MoeAccumulate)(
    const device bfloat* input,
    const device float* weights,
    device float* accumulator,
    device bfloat* output,
    constant uint& length,
    constant uint& weight_offset,
    constant uint& output_offset,
    constant uint& initialize,
    constant uint& finalize,
    constant uint& round_product,
    constant uint& round_accumulator,
    const uint position AXIS(length, 32)
) {
    float product = float(input[position]) * weights[weight_offset];
    if (round_product != 0) { product = float(bfloat(product)); }
    uint out = output_offset + position;
    float value = (initialize != 0 ? 0.0f : accumulator[out]) + product;
    if (round_accumulator != 0) { value = float(bfloat(value)); }
    accumulator[out] = value;
    if (finalize != 0) { output[out] = bfloat(value); }
}

PUBLIC KERNEL(MoeAdjustRoutes)(device uint* ids, device float* weights, const device float* scales,
    constant uint& rows,constant uint& experts,constant uint& active,constant uint& sort_ids,
    const uint row AXIS(rows,1)) {
    uint base=row*active;
    for(uint rank=0;rank<active;++rank) {
        uint id=ids[base+rank];
        weights[base+rank]=id<experts ? weights[base+rank]*scales[id] : NAN;
    }
    if(sort_ids!=0) for(uint rank=1;rank<active;++rank) {
        uint index=rank;
        while(index>0 && ids[base+index]<ids[base+index-1]) {
            uint id=ids[base+index];ids[base+index]=ids[base+index-1];ids[base+index-1]=id;
            float weight=weights[base+index];weights[base+index]=weights[base+index-1];weights[base+index-1]=weight;
            --index;
        }
    }
}

PUBLIC KERNEL(MoeSharedAdd)(
    const device bfloat* expert,
    const device bfloat* shared,
    const device bfloat* gate,
    device bfloat* output,
    constant uint& num_cols,
    constant uint& length,
    const uint position AXIS(length, 32)
) {
    float scale = float(bfloat(1.0f / (1.0f + exp(-float(gate[position / num_cols])))));
    float value = float(bfloat(float(shared[position]) * scale));
    output[position] = bfloat(float(expert[position]) + value);
}
