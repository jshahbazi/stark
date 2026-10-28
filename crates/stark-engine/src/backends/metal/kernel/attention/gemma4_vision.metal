#include <metal_stdlib>
#include "../common/dsl.h"

// Gemma 4 vision BF16 materialization boundaries. Independent implementation;
// references: MLX-VLM 53ebd1647cb83549b482a031ea70ba9d9ea60a04 (MIT), Google
// Gemma 4 E2B 3e22461f65e89153144f8adb70e3b8c2cc9845a7 (Apache-2.0).
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

PUBLIC KERNEL(Gemma4VisionClampBF16)(const device bfloat* input, device bfloat* output,
    constant uint& length, constant float& minimum, constant float& maximum,
    const uint i AXIS(length,128)) { output[i] = bfloat(clamp(float(input[i]),minimum,maximum)); }

PUBLIC KERNEL(Gemma4VisionNormBF16)(const device bfloat* input, const device bfloat* weight,
    device bfloat* output, constant uint& rows, constant uint& width, constant float& epsilon,
    constant uint& has_scale, const uint row AXIS(rows,32)) {
  float sum=0.0f;
  for(uint j=0;j<width;++j) { const float x=float(input[row*width+j]); sum += x*x; }
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
