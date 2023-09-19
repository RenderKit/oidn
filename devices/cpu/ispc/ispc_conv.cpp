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

    //ispc::CPUConvolutionKernel cKernel;
    //cKernel.src = toISPC<ispc::TensorAccessor3D>(*src);
    //cKernel.weight = toISPC<ispc::TensorAccessor4D>(*weight);
    //cKernel.bias = toISPC<ispc::TensorAccessor1D>(*bias);
    //cKernel.dst = toISPC<ispc::TensorAccessor3D>(*dst);
    const int blockC = getTensorLayoutInfo(dstDesc.layout).blockC;
    //ispc::CPUConvolutionKernel_Run(&cKernel);
    TensorAccessor4D<float, TensorLayout::OIhw8i8o> weightAccessor = *weight;
    TensorAccessor3D<float, TensorLayout::Chw8c> srcAccessor = *src;
    TensorAccessor3D<float, TensorLayout::Chw8c> dstAccessor = *dst;
    TensorAccessor1D<float> biasAccessor = *bias;
    std::stringstream sstream;
    sstream << "\nSrc CHW is " << src->getPaddedC() << "," << src->getH() << "," << src->getW() << std::endl;
    sstream << "Dst CHW is " << dst->getPaddedC() << "," << dst->getH() << "," << dst->getW() << std::endl;
    sstream << "Layout is SRC:" << static_cast<int>(src->getLayout()) << " DST: " << static_cast<int>(dst->getLayout()) << std::endl;
    sstream << "Weight OIHW is " << weightAccessor.O << "," << weightAccessor.I << "," << weightAccessor.H << "," << weightAccessor.W << std::endl;
    //std::cout << sstream.str();

    //for(int cb = 0; cb < dst->getPaddedC() / blockC; cb++)
    // Uncomment line above, and comment out line below (and L85 - bracket and colon) for single-threading
    parallel_nd(dst->getPaddedC() / blockC, [&](int cb)
    {
      // Incredibly slow, naive implementation, but here we go - right now, better correct than fast
      // Loop through the output channels
      for(int i = 0; i < blockC; i++)
      {
        int oc = (cb * blockC) + i;
        std::stringstream sstream;
        sstream << "Running for CHW: " << oc << "," << src->getH() << "," << src->getW() << std::endl;
        //std::cout << sstream.str();
        // Loop through output height
        for(int oh = 0; oh < src->getH(); oh++)
        {
          // Loop through output width
          for(int ow = 0; ow < src->getW(); ow++)
          {
            // Loop through the input channels
            for(int ic = 0; ic < src->getPaddedC(); ic++)
            {
              // Being in a blocked format means we have to retrieve every value manually
              for(int ih = (oh == 0 ? 0 : -1); ih <= (oh == (src->getH() - 1) ? 0 : 1); ih++)
              {
                for(int iw = (ow == 0 ? 0 : -1); iw <= (ow == (src->getW() - 1) ? 0 : 1); iw++)
                {
                  dstAccessor(oc, oh, ow) += (srcAccessor(ic, oh+ih, ow+iw) * ((weightAccessor(oc, ic, ih + 1, iw + 1))));
                }
              }
            }

            // Add the bias
            dstAccessor(oc, oh, ow) += biasAccessor(oc);

            // Activator
            dstAccessor(oc, oh, ow) = max(dstAccessor(oc, oh, ow), 0.0f);
          }
        }
      }
    }
    );
  }

OIDN_NAMESPACE_END