// Larger-Gemma vision only. Pinned PyTorch CPU FlashAttentionKernel.cpp and
// vec128_float_neon.h, 5c4886908584029761b579af026dcfb627c84070.
// BSD notice and attribution: scripts/licenses/distribution/.
#include <metal_stdlib>
#include "../common/dsl.h"
#include "gemma4_flash_source_exp_table.h"

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

static float gemma_flash_fast_exp(float x) {
  if(x < as_type<float>(0xc2af5dc1u)) return 0.0f;
  if(x > as_type<float>(0x42b0c0a5u)) return INFINITY;
  const float n=round(x*as_type<float>(0x3fb8aa3bu));
  const float r=fma(-n,as_type<float>(0x3f317218u),x);
  const float q=fma(r,as_type<float>(0x3e2ac976u),as_type<float>(0x3f00be9au));
  const float p=fma(q,r*r,1.0f+r);
  return as_type<float>(as_type<uint>(p)+(uint(int(n))<<23));
}

// Scalar rescaling has a separate precision policy from BF16 tile fexp.
// Compensated range reduction and degree-12 Taylor evaluation target nearest
// F32. This is measured against independent decimal and source-libm fixtures;
// it does not claim universally correctly rounded expf.
static float2 gemma_flash_pair_norm(float h,float l) {
  const float s=h+l;
  return float2(s,l-(s-h));
}
static float2 gemma_flash_pair_add(float2 a,float2 b) {
  const float s=a.x+b.x, v=s-a.x;
  const float e=(a.x-(s-v))+(b.x-v);
  return gemma_flash_pair_norm(s,e+(a.y+b.y));
}
static float2 gemma_flash_pair_mul(float2 a,float2 b) {
  const float h=a.x*b.x;
  const float l=fma(a.x,b.x,-h)+(a.x*b.y+a.y*b.x);
  return gemma_flash_pair_norm(h,l);
}
static float gemma_flash_round_scaled(float2 p,int k) {
  if(k > -126) return ldexp(p.x+p.y,k);
  const float h=ldexp(p.x,k+149), l=ldexp(p.y,k+149), integral=floor(h);
  const float2 tail=gemma_flash_pair_add(float2(h,l),float2(-integral,0.0f));
  const uint base=uint(integral);
  const bool up=tail.x>0.5f || (tail.x==0.5f && (tail.y>0.0f || (tail.y==0.0f && (base&1u))));
  return as_type<float>(base+uint(up));
}
static float gemma_flash_scalar_exp(float x) {
  // Caller supplies an old-maximum minus new-maximum: x <= 0, including -inf.
  if(x < -104.0f) return 0.0f;
  if(x == 0.0f) return 1.0f;
  const int k=int(rint(x*as_type<float>(0x3fb8aa3bu)));
  const float2 ln2=float2(as_type<float>(0x3f317218u),as_type<float>(0xb102e308u));
  const float2 r=gemma_flash_pair_add(float2(x,0.0f),gemma_flash_pair_mul(ln2,float2(-float(k),0.0f)));
  // Independent mathematical coefficients 1/n!, split into high/low F32.
  const float2 c[13]={
    float2(1.0f,0.0f),float2(1.0f,0.0f),float2(0.5f,0.0f),
    float2(as_type<float>(0x3e2aaaabu),as_type<float>(0xb1aaaaabu)),
    float2(as_type<float>(0x3d2aaaabu),as_type<float>(0xb0aaaaabu)),
    float2(as_type<float>(0x3c088889u),as_type<float>(0xafeeeeefu)),
    float2(as_type<float>(0x3ab60b61u),as_type<float>(0xae13e93fu)),
    float2(as_type<float>(0x39500d01u),as_type<float>(0xac3fcbfdu)),
    float2(as_type<float>(0x37d00d01u),as_type<float>(0xaabfcbfdu)),
    float2(as_type<float>(0x3638ef1du),as_type<float>(0x292ad8e6u)),
    float2(as_type<float>(0x3493f27eu),as_type<float>(0xa808760au)),
    float2(as_type<float>(0x32d7322bu),as_type<float>(0x25fea89cu)),
    float2(as_type<float>(0x310f76c7u),as_type<float>(0x24ff8d89u))};
  float2 p=c[12];
  for(int i=11;i>=0;--i) p=gemma_flash_pair_add(gemma_flash_pair_mul(p,r),c[i]);
  // Round in integer subnormal units before the bit cast, avoiding flush-to-zero
  // and a second subnormal arithmetic rounding at the underflow boundary.
  return gemma_flash_round_scaled(p,k);
}

// Independently derived numerical compatibility policy: a=ln2/128,
// 1+r*(a+a^3/32+r*a^2/2), with a mathematical 128-entry power table.
// The original nearest policy above remains an independent test control.
// Finite source controls establish compatibility, not host-source identity or
// universal bit-exactness (one broad strict-source counterexample is retained).
static float gemma_flash_source_exp(float x) {
  if(x < -104.0f) return 0.0f;
  if(x == 0.0f) return 1.0f;
  const float2 z=gemma_flash_pair_mul(GEMMA4_FLASH_SOURCE_INV,float2(x,0.0f));
  float k=rint(z.x);
  float2 r=gemma_flash_pair_add(z,float2(-k,0.0f));
  const bool odd=(int(k)&1)!=0;
  if(r.x>0.5f || (r.x==0.5f && (r.y>0.0f || (r.y==0.0f && odd)))) k+=1.0f;
  if(r.x< -0.5f || (r.x== -0.5f && (r.y<0.0f || (r.y==0.0f && odd)))) k-=1.0f;
  r=gemma_flash_pair_add(z,float2(-k,0.0f));
  const float2 p=gemma_flash_pair_add(gemma_flash_pair_mul(GEMMA4_FLASH_SOURCE_C0,r),GEMMA4_FLASH_SOURCE_C1);
  const float2 t=gemma_flash_pair_mul(p,r);
  const int index=int(k);
  const float2 scale=GEMMA4_FLASH_SOURCE_POW2[index&127];
  const float2 y=gemma_flash_pair_add(gemma_flash_pair_mul(t,scale),scale);
  return gemma_flash_round_scaled(y,index>>7);
}

PUBLIC KERNEL(Gemma4VisionFlashScalarExpF32)(const device float* input,device float* output,
    constant uint& count,const uint row AXIS(count,32)) {
  output[row]=gemma_flash_scalar_exp(input[row]);
}

PUBLIC KERNEL(Gemma4VisionFlashSourceExpF32)(const device float* input,device float* output,
    constant uint& count,const uint row AXIS(count,32)) {
  output[row]=gemma_flash_source_exp(input[row]);
}

// Integer RNE survives arithmetic optimization; its result additionally crosses
// a physical BF16 buffer into the separately dispatched PV pipeline.
static bfloat gemma_flash_store_bf16(float x) {
  const uint bits=as_type<uint>(x);
  const ushort rounded=ushort((bits+0x7fffu+((bits>>16)&1u))>>16);
  return as_type<bfloat>(rounded);
}

PUBLIC KERNEL(Gemma4VisionFlashQK)(const device bfloat* queries,const device bfloat* keys,
    device float* scores,constant uint& query_rows,constant uint& key_rows,
    constant uint& heads,constant uint& key_start,const uint i AXIS(heads*query_rows*512,32)) {
  const uint column=i%512, plane=i/512, head=plane/query_rows;
  const uint key_row=key_start+column;
  if(key_row>=key_rows) { scores[i]=-INFINITY; return; }
  const uint query=plane*128, key=(key_row*heads+head)*128;
  // F32-output CPUBlas gemm_no_downcast follows the four-part generic sum,
  // not the 32-stream BF16-output Linear specialization.
  float4 sums=0.0f;
  for(uint start=0;start<72;start+=4)
    for(uint lane=0;lane<4;++lane) {
      const uint channel=start+lane;
      sums[lane]=fma(float(queries[query+channel]),float(keys[key+channel]),sums[lane]);
    }
  scores[i]=((sums[0]+sums[1])+sums[2])+sums[3];
}

PUBLIC KERNEL(Gemma4VisionFlashExpBF16)(const device float* scores,device bfloat* exponents,
    device float* state,constant uint& query_rows,constant uint& heads,constant uint& key_start,
    const uint row AXIS(heads*query_rows,32)) {
  const uint base=row*512;
  const float previous_max=key_start==0 ? -INFINITY : state[row*3];
  const float previous_sum=key_start==0 ? 0.0f : state[row*3+1];
  float maximum=previous_max;
  for(uint column=0;column<512;++column) maximum=max(maximum,scores[base+column]);
  const float factor=gemma_flash_source_exp(previous_max-maximum);
  float partial[8]={};
  for(uint column=0;column<512;++column) {
    const float value=gemma_flash_fast_exp(scores[base+column]-maximum);
    partial[column%8]+=value;
    exponents[base+column]=gemma_flash_store_bf16(value);
  }
  const float4 lanes=float4(partial[0]+partial[4],partial[1]+partial[5],partial[2]+partial[6],partial[3]+partial[7]);
  const float total=(lanes[0]+lanes[2])+(lanes[1]+lanes[3]);
  state[row*3]=maximum; state[row*3+1]=fma(factor,previous_sum,total); state[row*3+2]=factor;
}

// Test-only dispatch: preserve the ordinary finish reciprocal in F32 for audit.
PUBLIC KERNEL(Gemma4VisionFlashReciprocalF32)(const device float* state,device float* inverse,
    constant uint& query_rows,constant uint& heads,const uint row AXIS(heads*query_rows,32)) {
  inverse[row]=1.0f/state[row*3+1];
}

PUBLIC KERNEL(Gemma4VisionFlashPVF32)(const device bfloat* exponents,const device bfloat* values,
    const device float* state,device float* accumulator,constant uint& query_rows,
    constant uint& key_rows,constant uint& heads,constant uint& key_start,
    const uint i AXIS(heads*query_rows*72,32)) {
  const uint channel=i%72, plane=i/72, head=plane/query_rows;
  float4 sums=0.0f;
  for(uint column=0;column<512;++column) {
    const uint key=key_start+column;
    if(key<key_rows)
      sums[column%4]=fma(float(exponents[plane*512+column]),float(values[(key*heads+head)*128+channel]),sums[column%4]);
  }
  const float current=((sums[0]+sums[1])+sums[2])+sums[3];
  const float previous=key_start==0 ? 0.0f : accumulator[i]*state[plane*3+2];
  accumulator[i]=previous+current;
}

PUBLIC KERNEL(Gemma4VisionFlashFinishBF16)(const device float* accumulator,const device float* state,
    device bfloat* output,constant uint& query_rows,constant uint& heads,
    const uint i AXIS(heads*query_rows*72,128)) {
  const uint channel=i%72, plane=i/72, head=plane/query_rows, row=plane%query_rows;
  const float inverse=1.0f/state[plane*3+1];
  output[(row*heads+head)*72+channel]=gemma_flash_store_bf16(accumulator[i]*inverse);
}
