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

export void CPUConvolutionKernel_Run(uniform CPUConvolutionKernel* uniform self)
{
    // Do convolution stuff here.
    // For now, print out to ensure data has been passed correctly.

    print("We hit the ISPC conv kernel successfully! We have the following sizes:\n");
    print("SRC: C=% H=% W=%\n", self->src.C, self->src.H, self->src.W);
    print("WEI: O=% I=% H=% W=%\n", self->weight.O, self->weight.I, self->weight.H, self->weight.W);
    print("BIA: X=%\n", self->bias.X);
    print("DST: C=% H=% W=%\n", self->dst.C, self->dst.H, self->dst.W);
}