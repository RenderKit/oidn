#include "tensor_accessor_1d.isph"
#include "tensor_accessor_4d.isph"
#include "../tensor_accessor_3d.isph"

struct CPUConvolutionKernel
{
  uniform TensorAccessor3D src;
  uniform TensorAccessor4D weight;
  uniform TensorAccessor1D bias;
  uniform TensorAccessor3D dst;
};

export void CPUConvolutionKernel_Run(uniform CPUConvolutionKernel* uniform self, uniform int c)
{
    const uniform size_t W = self->dst.W;
    const uniform size_t H = self->dst.H;
    const uniform size_t srcC = self->src.C;

    // value(n, c, h, w) = n * CHW + c * HW + h * W + w
    const uniform size_t dstOffset = c*H*W;
    uniform float* const uniform dstPtr = self->dst.ptr + dstOffset;

    for(uniform int oh = 0; oh < H; oh++)
    {
      for(uniform int ow = 0; ow < W; ow++)
      {
        varying float value = 0.0f;

        for(uniform int ic = 0; ic < srcC; ic++)
        {
          for(uniform int ih = (oh == 0 ? 0 : -1); ih <= (oh == (H-1) ? 0 : 1); ih++)
          {
            for(uniform int iw = (ow == 0 ? 0 : -1); iw <= (ow == (W - 1) ? 0 : 1); iw++)
            {
              value += (get1f(self->src, ic, oh+ih, ow+iw) * get1f(self->weight, c, ic, ih+1, iw+1));
            }
          }
        }

        // Bias
        value += self->bias.ptr[c];

        // Activator
        value = max(value, 0.0f);

        // Store it non-temporally
        streaming_store(&dstPtr[((oh)*W) + (ow)], value);
      }
    }
}