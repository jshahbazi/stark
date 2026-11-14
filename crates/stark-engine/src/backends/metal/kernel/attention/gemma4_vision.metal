#include <metal_stdlib>
#include "../common/dsl.h"

// Gemma 4 vision BF16 materialization boundaries. Independent implementation;
// references: MLX-VLM 53ebd1647cb83549b482a031ea70ba9d9ea60a04 (MIT), Google
// Gemma 4 E2B 3e22461f65e89153144f8adb70e3b8c2cc9845a7 (Apache-2.0).
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

PUBLIC KERNEL(Gemma4UnifiedPositionBF16)(const device bfloat* input,const device bfloat* positions,
    device bfloat* output,constant uint& rows,constant uint& width,constant uint& grid_width,
    const uint i AXIS(rows*width,128)) {
    const uint row=i/width, channel=i%width, x=row%grid_width, y=row/grid_width;
    const float position=float(bfloat(float(positions[x*2*width+channel])+float(positions[(y*2+1)*width+channel])));
    output[i]=bfloat(float(input[i])+position);
}

PUBLIC KERNEL(Gemma4VisionClampBF16)(const device bfloat* input, device bfloat* output,
    constant uint& length, constant float& minimum, constant float& maximum,
    const uint i AXIS(length,128)) { output[i] = bfloat(clamp(float(input[i]),minimum,maximum)); }

PUBLIC KERNEL(Gemma4VisionNormBF16)(const device bfloat* input, const device bfloat* weight,
    device bfloat* output, constant uint& rows, constant uint& width, constant float& epsilon,
    constant uint& has_scale, const uint row AXIS(rows,32)) {
  float sum=0.0f;
  if(width==72 || width==1152) {
    // Larger Gemma vision only. ATen cascade sum; see CPU implementation and
    // scripts/licenses/distribution/pytorch-LICENSE. Logical72 has an 8-tail.
    const uint count=width/16;
    const uint power=max(4u,(32u-clz(count-1))/4u), step=1u<<power, mask=step-1;
    float4 levels[4][4]={};
    uint index=0;
    while(index+step<=count) {
      for(uint j=0;j<step;++j,++index)
        for(uint reg=0;reg<4;++reg)
          for(uint lane=0;lane<4;++lane) {
            const float x=float(input[row*width+index*16+reg*4+lane]);
            levels[0][reg][lane]+=x*x;
          }
      for(uint level=1;level<4;++level) {
        for(uint reg=0;reg<4;++reg) { levels[level][reg]+=levels[level-1][reg]; levels[level-1][reg]=0.0f; }
        if((index&(mask<<(level*power)))!=0) break;
      }
    }
    for(;index<count;++index)
      for(uint reg=0;reg<4;++reg)
        for(uint lane=0;lane<4;++lane) {
          const float x=float(input[row*width+index*16+reg*4+lane]);
          levels[0][reg][lane]+=x*x;
        }
    for(uint level=1;level<4;++level)
      for(uint reg=0;reg<4;++reg) levels[0][reg]+=levels[level][reg];
    for(uint tail=count*16;tail<width;tail+=4)
      for(uint lane=0;lane<4;++lane) {
        const float x=float(input[row*width+tail+lane]);
        levels[0][0][lane]+=x*x;
      }
    float4 lanes=levels[0][0];
    for(uint reg=1;reg<4;++reg) lanes+=levels[0][reg];
    for(uint lane=0;lane<4;++lane) sum+=lanes[lane];
  } else {
    for(uint j=0;j<width;++j) { const float x=float(input[row*width+j]); sum += x*x; }
  }
  const float scale=1.0f/metal::precise::sqrt(sum/float(width)+epsilon);
  for(uint j=0;j<width;++j) {
    const float x=float(input[row*width+j])*scale;
    output[row*width+j]=bfloat(x*(has_scale==0 ? 1.0f : float(weight[j])));
  }
}

PUBLIC KERNEL(Gemma4VisionPrepareBF16)(const device bfloat* q,const device bfloat* k,const device bfloat* v,
    device bfloat* queries,device bfloat* keys,device bfloat* values,
    const device bfloat* cosines,const device bfloat* sines,constant uint& rows,
    constant uint& num_heads,constant uint& head_dim,const uint i AXIS(rows*num_heads*head_dim,128)) {
  const uint row=i/(num_heads*head_dim), head=(i/head_dim)%num_heads, channel=i%head_dim;
  const uint aw=head_dim/2, half_width=aw/2, local=channel%aw;
  const uint paired=(channel/aw)*aw+(local+half_width)%aw, base=(row*num_heads+head)*head_dim;
  const float cosine=float(cosines[row*head_dim+channel]), sine=float(sines[row*head_dim+channel]);
  const float sign=local<half_width ? -1.0f : 1.0f;
  const float qa=float(bfloat(float(q[i])*cosine));
  const float qb=float(bfloat(sign*float(q[base+paired])*sine));
  const float ka=float(bfloat(float(k[i])*cosine));
  const float kb=float(bfloat(sign*float(k[base+paired])*sine));
  queries[(head*rows+row)*head_dim+channel]=bfloat(qa+qb);
  keys[i]=bfloat(ka+kb); values[i]=v[i];
}

PUBLIC KERNEL(Gemma4VisionProductBF16)(const device bfloat* left,const device bfloat* right,
    device bfloat* output,constant uint& length,const uint i AXIS(length,128)) {
  output[i]=bfloat(float(left[i])*float(right[i]));
}

PUBLIC KERNEL(Gemma4VisionPoolBF16)(const device bfloat* input,device bfloat* output,
    constant uint& grid_width,constant uint& grid_height,constant uint& width,
    const uint i AXIS((grid_width/3)*(grid_height/3)*width,128)) {
  const uint channel=i%width, x=(i/width)%(grid_width/3),y=i/(width*(grid_width/3));
  float sum=0.0f;
  for(uint dy=0;dy<3;++dy) for(uint dx=0;dx<3;++dx)
    sum += float(input[((y*3+dy)*grid_width+x*3+dx)*width+channel])/9.0f;
  output[i]=bfloat(float(bfloat(sum))*metal::precise::sqrt(float(width)));
}

PUBLIC KERNEL(Gemma4VisionPreparePaddedBF16)(const device bfloat* q,const device bfloat* k,const device bfloat* v,
    device bfloat* queries,device bfloat* keys,device bfloat* values,
    const device bfloat* cosines,const device bfloat* sines,constant uint& rows,
    constant uint& num_heads,constant uint& head_dim,constant uint& padded_dim,
    const uint i AXIS(rows*num_heads*padded_dim,128)) {
  const uint row=i/(num_heads*padded_dim), head=(i/padded_dim)%num_heads, channel=i%padded_dim;
  const uint query_index=(head*rows+row)*padded_dim+channel;
  if(channel>=head_dim) { queries[query_index]=bfloat(0); keys[i]=bfloat(0); values[i]=bfloat(0); return; }
  const uint aw=head_dim/2, half_width=aw/2, local=channel%aw;
  const uint paired=(channel/aw)*aw+(local+half_width)%aw, base=(row*num_heads+head)*head_dim;
  const float cosine=float(cosines[row*head_dim+channel]), sine=float(sines[row*head_dim+channel]);
  const float sign=local<half_width ? -1.0f : 1.0f;
  const float qa=float(bfloat(float(q[base+channel])*cosine));
  const float qb=float(bfloat(sign*float(q[base+paired])*sine));
  const float ka=float(bfloat(float(k[base+channel])*cosine));
  const float kb=float(bfloat(sign*float(k[base+paired])*sine));
  queries[query_index]=bfloat(qa+qb); keys[i]=bfloat(ka+kb); values[i]=v[base+channel];
}

PUBLIC KERNEL(Gemma4VisionUnpadBF16)(const device bfloat* input,device bfloat* output,
    constant uint& length,constant uint& head_dim,constant uint& padded_dim,const uint i AXIS(length,128)) {
  output[i]=input[(i/head_dim)*padded_dim+i%head_dim];
}

PUBLIC KERNEL(Gemma4VisionPoolStandardizedBF16)(const device bfloat* input,const device bfloat* bias,
    const device bfloat* scale,device bfloat* output,constant uint& grid_width,constant uint& grid_height,
    constant uint& width,const uint i AXIS((grid_width/3)*(grid_height/3)*width,128)) {
  const uint channel=i%width, x=(i/width)%(grid_width/3),y=i/(width*(grid_width/3));
  float sum=0.0f;
  for(uint dy=0;dy<3;++dy) for(uint dx=0;dx<3;++dx)
    sum += float(input[((y*3+dy)*grid_width+x*3+dx)*width+channel])/9.0f;
  const float pooled=float(bfloat(sum))*metal::precise::sqrt(float(width));
  output[i]=bfloat((pooled-float(bias[channel]))*float(scale[channel]));
}
