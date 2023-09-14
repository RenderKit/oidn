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
    cKernel.dst = toISPC<ispc::TensorAccessor3D>(*dst);
    const int blockC = getTensorLayoutInfo(dstDesc.layout).blockC;
    //ispc::CPUConvolutionKernel_Run(&cKernel);
    std::stringstream sstream;
    sstream << "\nSrc C is " << src->getPaddedC() << std::endl;
    sstream << "Dst C is " << dst->getPaddedC() << " with blockC " << blockC << std::endl;
    sstream << "Layout is SRC:" << static_cast<int>(src->getLayout()) << " DST: " << static_cast<int>(dst->getLayout()) << std::endl;
    std::cout << sstream.str();
    float* dstData = static_cast<float *>(dst->getData());
    float* srcData = static_cast<float *>(src->getData());
    float* weightData = static_cast<float *>(weight->getData());
    float* biasData = static_cast<float *>(bias->getData());

    // Uncomment line below, and comment out line below that for single-threading
    //for(int cb = 0; cb < dst->getPaddedC() / blockC; cb++)
    // Uncomment line above, and comment out line below (and L85 - bracket and colon) for single-threading
    parallel_nd(dst->getPaddedC() / blockC, [&](int cb)
    {
      // Taken from https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html#blocked-layout
      auto getIndexOf = [=](std::shared_ptr<Tensor> tensor, int tc, int th, int tw) -> size_t {
        return ((tc/8) * tensor->getH() * tensor->getW() * 8) + (th * tensor->getW() * 8) + (tw * 8) + (tc % 8);
      };

      auto getIndexOfWeight = [=](int wo, int wi, int wh, int ww) -> size_t {
        return ((((((size_t)weight->getPaddedI() * wo) + wi) * (size_t)weight->getH()) + wh) * (size_t)weight->getW()) + ww;
      };

      // Incredibly slow, naive implementation, but here we go - right now, better correct than fast
      // Loop through the output channels
      for(int i = 0; i < blockC; i++)
      {
        int oc = (cb * blockC) + i;
        std::stringstream sstream;
        sstream << "Running for CHW: " << oc << "," << src->getH() << "," << src->getW() << std::endl;
        std::cout << sstream.str();
        //TODO: handle edge cases, for now we inset 1px so we have all pixels populated
        // Loop through output height
        for(int oh = 1; oh < src->getH() - 1; oh++)
        {
          // Loop through output width
          for(int ow = 1; ow < src->getW() - 1; ow++)
          {
            float* dstPixel = &dstData[getIndexOf(dst, oc, oh, ow)];
            *dstPixel = 0;
            // Loop through the input channels
            for(int ic = 0; ic < src->getPaddedC(); ic++)
            {
              // Calculate a pointer for the start of the 3x3 filter
              float* filter = &weightData[getIndexOfWeight(oc, ic, 0, 0)];

              // Being in a blocked format means we have to retrieve every value manually
              for(int ih = -1; ih <= 1; ih++)
              {
                for(int iw = -1; iw <= 1; iw++)
                {
                  float inPixel = srcData[getIndexOf(src, ic, oh+ih, ow+iw)];
                  *dstPixel += (inPixel * filter[((ih+1)*3) + (iw + 1)]);
                }
              }
            }

            *dstPixel += biasData[oc];
          }
        }
      }
    }
    );
  }

OIDN_NAMESPACE_END