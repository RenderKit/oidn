#include "../core/node.h"

#pragma once

#if defined(USE_BNNS)
#include <Accelerate/Accelerate.h>

namespace oidn
{
class BnnsNode : public Node
{
public:
    std::shared_ptr<memory> getDst() const override { return dst; }
    void execute(stream *) override
    {
        float *s = (float *)src->map_data();
        float *d = (float *)dst->map_data();
        BNNSFilterApply(filter, s, d);
        src->unmap_data(s);
        dst->unmap_data(d);
    }

    BnnsNode(std::shared_ptr<memory> asrc, std::shared_ptr<memory>adst) : src(asrc),dst(adst),filter(nullptr) {}
    ~BnnsNode() override
    {
        if (filter) BNNSFilterDestroy(filter);
    }

protected:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;
    BNNSFilter filter;
};

class BnnsConvNode : public BnnsNode
{
public:
    BnnsConvNode(BNNSLayerParametersConvolution desc,std::shared_ptr<memory> asrc, std::shared_ptr<memory>adst) :
    BnnsNode(asrc,adst)
    {
        filter = BNNSFilterCreateLayerConvolution(&desc,nullptr);
        if (!filter) throw Exception(Error::Unknown, "BNNS Convolution creation failed");
    }
};


class BnnsUpsampleNode : public BnnsNode
{
private:
public:
    BnnsUpsampleNode(
                 const std::shared_ptr<memory>& src,
                 const std::shared_ptr<memory>& dst)
    : BnnsNode(src,dst)
  {  }
    void execute(stream *) override
    {
        memory::dims srcDims = getMemoryDims(src);
        size_t C = srcDims[1];
        size_t H = srcDims[2];
        size_t W = srcDims[3];
  
        float *pSrc = (float*)src->map_data();
        float *pDst = (float*)dst->map_data();

        parallel_nd(C, [&](int c)
        {
            for (size_t h=0;h<H;h++)
            {
                #pragma unroll(32)
                for (size_t w=0;w<W;w++)
                {
                    float val = pSrc[c*H*W + h*W + w];
                    pDst[c*H*W*4 + 4*h*W + 2*w] = val;
                    pDst[c*H*W*4 + 4*h*W + 2*w + 1] = val;
                    pDst[c*H*W*4 + 4*h*W + 2*W + 2*w] = val;
                    pDst[c*H*W*4 + 4*h*W + 2*W + 2*w + 1] = val;
                }
            }
        });
        src->unmap_data(pSrc);
        dst->unmap_data(pDst);
    }
};



class BnnsPoolNode : public BnnsNode
{
private:
public:
    BnnsPoolNode(BNNSLayerParametersPooling desc,
                 const std::shared_ptr<memory>& src,
                 const std::shared_ptr<memory>& dst)
    : BnnsNode(src,dst)
  {
      filter = BNNSFilterCreateLayerPooling(&desc,nullptr);
      if (!filter) throw Exception(Error::Unknown, "BNNS pool creation failed");
  }
};



class BnnsInputReorderNode : public InputReorderNode
{
public:
    BnnsInputReorderNode(const Image& srcColor,
                     const Image& srcAlbedo,
                     const Image& srcNormal,
                     const std::shared_ptr<memory>& dst,
                     const std::shared_ptr<TransferFunction>& transferFunc,
                     bool hdr)
                        :
                        InputReorderNode(srcColor,srcAlbedo,srcNormal,dst,transferFunc,hdr)
                        {}
 
 
    void execute(stream *) override
    {
      assert(data.H + data.hSrcBegin <= srcColor.height);
      assert(data.W + data.wSrcBegin <= srcColor.width);
      assert(data.H + data.hDstBegin <= data.dst.H);
      assert(data.W + data.wDstBegin <= data.dst.W);

      parallel_nd(data.dst.H, [&](int hDst)
      {
        ispc::InputReorder_kernel_chw(&data, hDst);
      });
    }
};




class BnnsOutputReorderNode : public OutputReorderNode
{

public:
    BnnsOutputReorderNode(const std::shared_ptr<memory>& src,
                          const Image& dst,
                          const std::shared_ptr<TransferFunction>& transferFunc,
                          bool hdr)
    : OutputReorderNode(src,dst,transferFunc,hdr)
    { }


  void execute(stream *) override
  {
    assert(data.hSrcBegin + data.H <= data.src.H);
    assert(data.wSrcBegin + data.W <= data.src.W);

    parallel_nd(data.H, [&](int h)
    {
      ispc::OutputReorder_kernel_chw(&data, h);
    });
  }
};

};


#endif
