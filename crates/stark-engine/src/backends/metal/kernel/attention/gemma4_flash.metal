// Larger-Gemma vision only. Pinned PyTorch CPU FlashAttentionKernel.cpp and
// vec128_float_neon.h, 5c4886908584029761b579af026dcfb627c84070.
// BSD notice and attribution: scripts/licenses/distribution/.
#include <metal_stdlib>
#include "../common/dsl.h"

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
  const uint column=i%512, plane=i/512, head=plane/query_rows, row=plane%query_rows;
  const uint key_row=key_start+column;
  if(key_row>=key_rows) { scores[i]=-INFINITY; return; }
  const uint query=plane*128, key=(key_row*heads+head)*128;
  float4 sums[8]={};
  for(uint start=0;start<64;start+=32)
    for(uint reg=0;reg<8;++reg)
      for(uint lane=0;lane<4;++lane) {
        const uint channel=start+reg*4+lane;
        sums[reg][lane]=fma(float(queries[query+channel]),float(keys[key+channel]),sums[reg][lane]);
      }
  for(uint offset=4;offset!=0;offset>>=1)
    for(uint reg=0;reg<offset;++reg) sums[reg]=sums[reg]+sums[reg+offset];
  float4 tail=0.0f;
  for(uint part=0;part<2;++part)
    for(uint lane=0;lane<4;++lane) {
      const uint channel=64+part*4+lane;
      tail[lane]=fma(float(queries[query+channel]),float(keys[key+channel]),tail[lane]);
    }
  const float dot=(sums[0][0]+sums[0][1])+(sums[0][2]+sums[0][3]);
  scores[i]=dot+((tail[0]+tail[1])+(tail[2]+tail[3]));
}

PUBLIC KERNEL(Gemma4VisionFlashExpBF16)(const device float* scores,device bfloat* exponents,
    device float* state,constant uint& query_rows,constant uint& heads,constant uint& key_start,
    const uint row AXIS(heads*query_rows,32)) {
  const uint base=row*512;
  const float previous_max=key_start==0 ? -INFINITY : state[row*3];
  const float previous_sum=key_start==0 ? 0.0f : state[row*3+1];
  float maximum=previous_max;
  for(uint column=0;column<512;++column) maximum=max(maximum,scores[base+column]);
  const float factor=metal::precise::exp(previous_max-maximum);
  float partial[8]={};
  for(uint column=0;column<512;++column) {
    const float value=gemma_flash_fast_exp(scores[base+column]-maximum);
    partial[column%8]+=value;
    exponents[base+column]=gemma_flash_store_bf16(value);
  }
  const float4 lanes=float4(partial[0]+partial[4],partial[1]+partial[5],partial[2]+partial[6],partial[3]+partial[7]);
  const float total=(lanes[0]+lanes[2])+(lanes[1]+lanes[3]);
  state[row*3]=maximum; state[row*3+1]=total+factor*previous_sum; state[row*3+2]=factor;
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
