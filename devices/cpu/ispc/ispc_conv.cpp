#include "ispc_conv.h"
#include "cpu_convolution_ispc.h"

OIDN_NAMESPACE_BEGIN

  ISPCConv::ISPCConv(const Ref<ISPCEngine>& engine, const ConvDesc& desc)
    : Conv(desc), engine(engine)
  { }

  void ISPCConv::submit()
  {
    if (!src || !dst || !weight || !bias)
      throw std::logic_error("convolution source/weight/bias/destination not set");

    ispc::CPUConvolutionKernel cKernel;
    cKernel.src = toISPC<ispc::TensorAccessor3D>(*src);
    cKernel.weight = toISPC<ispc::TensorAccessor4D>(*weight);
    cKernel.bias = toISPC<ispc::TensorAccessor1D>(*bias);
    cKernel.dst = toISPC<ispc::TensorAccessor3D>(*dst);;

    // Loop through the output channels (using TBB)
    //for(int oc = 0; oc < dst->getPaddedC(); oc++)
    // Uncomment line above, and comment out line below (and L60 - bracket and colon) for single-threading
    parallel_nd(dst->getPaddedC(), [&](int oc)
    {
      ispc::CPUConvolutionKernel_Run(&cKernel, oc);
    }
    );
  }

OIDN_NAMESPACE_END