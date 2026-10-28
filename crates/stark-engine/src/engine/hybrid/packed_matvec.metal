// GGML quantization layout adaptation, MIT. See LICENSE.ggml-hybrid.
// Lookup tables are supplied from the same source-bound Rust constants.

struct HybridMatrixParams { uint rows, columns, kind, block, block_bytes, bits, group, scale_kind; };

inline ushort hybrid_u16(device const uchar *p) { return ushort(p[0]) | (ushort(p[1]) << 8); }
inline uint hybrid_u32(device const uchar *p) { return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24); }
inline float hybrid_half(device const uchar *p) { return float(as_type<half>(hybrid_u16(p))); }
inline float hybrid_float(device const uchar *p, uint kind) {
    if (kind == 0) return as_type<float>(hybrid_u32(p));
    if (kind == 30) return as_type<float>(uint(hybrid_u16(p)) << 16);
    return hybrid_half(p);
}
inline uint2 hybrid_scale_min(device const uchar *s, uint i) {
    if (i < 4) return uint2(s[i] & 63, s[i + 4] & 63);
    return uint2((s[i+4]&15) | ((s[i-4]>>6)<<4), (s[i+4]>>4) | ((s[i]>>6)<<4));
}
inline float hybrid_quant(device const uchar *p, uint kind, uint i) {
    if (kind == 0 || kind == 1 || kind == 30) return hybrid_float(p, kind);
    if (kind == 8) return hybrid_half(p) * float(as_type<char>(p[2+i]));
    if (kind == 10 || kind == 11) {
        uint n=i/128, j=(i%128)/32, h=(i%32)/16, l=i%16;
        if (kind == 10) {
            uint s=p[n*8+j*2+h];
            uint q=(p[16+n*32+h*16+l]>>(2*j))&3;
            return hybrid_half(p+80)*float(s&15)*float(q)-hybrid_half(p+82)*float(s>>4);
        }
        uint a=hybrid_u32(p+96), b=hybrid_u32(p+100), c=hybrid_u32(p+104);
        uint packed[4] = { (a&0x0f0f0f0f)|((c&0x03030303)<<4),
            (b&0x0f0f0f0f)|(((c>>2)&0x03030303)<<4),
            ((a>>4)&0x0f0f0f0f)|(((c>>4)&0x03030303)<<4),
            ((b>>4)&0x0f0f0f0f)|(((c>>6)&0x03030303)<<4) };
        uint si=n*8+j*2+h;
        int scale=int((packed[si/4]>>(8*(si%4)))&255)-32;
        int q=int((p[32+n*32+h*16+l]>>(2*j))&3)-((p[h*16+l]&(1<<(n*4+j))) ? 0 : 4);
        return hybrid_half(p+108)*float(scale)*float(q);
    }
    if (kind == 12 || kind == 13) {
        uint part=i/64, h=(i%64)/32, l=i%32;
        uint2 sm=hybrid_scale_min(p+4, part*2+h);
        uint q=(p[(kind==13?48:16)+part*32+l]>>(h*4))&15;
        if (kind==13 && (p[16+l]&(1<<(part*2+h)))) q+=16;
        return hybrid_half(p)*float(sm.x)*float(q)-hybrid_half(p+2)*float(sm.y);
    }
    if (kind == 14) {
        uint n=i/128, part=(i%128)/32, l=i%32;
        uint ql=p[n*64+(part%2)*32+l], qh=p[128+n*32+l];
        int q=int(((ql>>(part>=2?4:0))&15)|(((qh>>(part*2))&3)<<4))-32;
        return hybrid_half(p+208)*float(as_type<char>(p[192+n*8+l/16+part*2]))*float(q);
    }
    if (kind == 16) {
        uint part=i/32, l=(i%32)/8, j=i%8;
        uint at=2+part*8, ss=hybrid_u32(p+at+4);
        ulong grid=iq2xxs_grid[p[at+l]];
        uint signs=ksigns[(ss>>(7*l))&127];
        return hybrid_half(p)*(0.5f+float(ss>>28))*0.25f*float((grid>>(8*j))&255ul)*((signs&(1<<j))?-1.0f:1.0f);
    }
    if (kind == 17) {
        uint part=i/32, l=(i%32)/8, j=i%8;
        uint q=hybrid_u16(p+2+part*8+l*2), scales=p[66+part];
        ulong grid=iq2xs_grid[q&511];
        uint signs=ksigns[q>>9];
        return hybrid_half(p)*(0.5f+float((scales>>(l<2?0:4))&15))*0.25f*float((grid>>(8*j))&255ul)*((signs&(1<<j))?-1.0f:1.0f);
    }
    if (kind == 18) {
        uint part=i/32, l=(i%32)/8, j=i%8;
        uint ss=hybrid_u32(p+66+part*4);
        uint signs=ksigns[(ss>>(7*l))&127];
        uint grid=iq3xxs_grid[p[2+part*8+2*l+(j>=4?1:0)]];
        return hybrid_half(p)*(0.5f+float(ss>>28))*0.5f*float((grid>>(8*(j%4)))&255u)*((signs&(1<<j))?-1.0f:1.0f);
    }
    // IQ4_XS (type 23).
    uint part=i/32, j=i%16;
    uint low=(p[4+part/2]>>(4*(part%2)))&15;
    int scale=int(low|(((hybrid_u16(p+2)>>(part*2))&3)<<4))-32;
    uint q=p[8+part*16+j];
    return hybrid_half(p)*float(scale)*float(iq4values[(q>>((i%32)>=16?4:0))&15]);
}

inline void hybrid_write_sum(float sum, device float *output, uint row,
    uint lane, uint simd_lane, uint simd_group, threadgroup float *partial) {
    float total=simd_sum(sum);
    if (simd_lane==0) partial[simd_group]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane==0) { float result=0; for (uint i=0;i<8;++i) result+=partial[i]; output[row]=result; }
}

kernel void hybrid_packed_matvec(device const uchar *codes [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]],
    constant HybridMatrixParams &p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    float sum=0;
    // Packed matrices use the otherwise-unused group field as rows per batch.
    ulong input_offset=p.group ? ulong(row/p.group)*p.columns : 0;
    for (uint col=lane;col<p.columns;col+=256) {
        ulong block=(ulong(row)*p.columns+col)/p.block;
        sum+=hybrid_quant(codes+block*p.block_bytes,p.kind,col%p.block)*input[input_offset+col];
    }
    hybrid_write_sum(sum,output,row,lane,simd_lane,simd_group,partial);
}

kernel void hybrid_affine_matvec(device const uchar *codes [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]],
    constant HybridMatrixParams &p [[buffer(3)]], device const uchar *scales [[buffer(4)]],
    device const uchar *biases [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    uint item=p.scale_kind==0?4:2;
    float sum=0;
    // Affine matrices use the otherwise-unused block field as rows per batch.
    ulong input_offset=p.block ? ulong(row/p.block)*p.columns : 0;
    for (uint col=lane;col<p.columns;col+=256) {
        ulong element=ulong(row)*p.columns+col;
        uint q=(codes[element*p.bits/8]>>((col*p.bits)%8))&((1u<<p.bits)-1);
        ulong group=element/p.group;
        float value=hybrid_float(scales+group*item,p.scale_kind)*float(q)+hybrid_float(biases+group*item,p.scale_kind);
        sum+=value*input[input_offset+col];
    }
    hybrid_write_sum(sum,output,row,lane,simd_lane,simd_group,partial);
}
