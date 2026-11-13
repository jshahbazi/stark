// Original Stark test dispatch over the independently qualified packed decoder.
// Quantization conventions retain packed_matvec.metal's GGML MIT attribution.
kernel void hybrid_edge0_block_decode_reference(
    device const uchar *codes [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint4 &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.y * p.z) return;
    output[i] = hybrid_quant(codes + ulong(i / p.z) * p.w, p.x, i % p.z);
}
