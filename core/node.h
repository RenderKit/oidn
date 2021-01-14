// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor.h"
#include <vector>

namespace oidn {

  class Node : public RefCount
  {
  public:
    virtual ~Node() = default;

    virtual void execute(stream& sm) = 0;

    virtual Ref<Tensor> getDst() const { return nullptr; }

    virtual size_t getScratchpadSize() const { return 0; }
    virtual void setScratchpad(const Ref<Tensor>& scratchpad) {}

    virtual void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W)
    {
      assert(0); // not supported
    }
  };

  // Node wrapping an MKL-DNN primitive
  class MklNode : public Node
  {
  private:
    primitive prim;
    std::unordered_map<int, memory> args;
    Ref<Tensor> scratchpad;

  public:
    MklNode(const primitive& prim, const std::unordered_map<int, memory>& args)
      : prim(prim),
        args(args)
    {}

    size_t getScratchpadSize() const override
    {
      const auto primDesc = prim.get_primitive_desc();
      const dnnl_memory_desc_t* scratchpadDesc = dnnl_primitive_desc_query_md(primDesc, dnnl_query_scratchpad_md, 0);
      if (scratchpadDesc == nullptr)
        return 0;
      return dnnl_memory_desc_get_size(scratchpadDesc);
    }

    void setScratchpad(const Ref<Tensor>& scratchpad) override
    {
      this->scratchpad = scratchpad;
      args.insert(std::make_pair(DNNL_ARG_SCRATCHPAD, scratchpad->mem));
    }

    void execute(stream& sm) override
    {
      prim.execute(sm, args);
    }
  };

  // Convolution node
  class ConvNode : public MklNode
  {
  private:
    Ref<Tensor> src;
    Ref<Tensor> weights;
    Ref<Tensor> bias;
    Ref<Tensor> dst;

  public:
    ConvNode(const convolution_forward::primitive_desc& desc,
             const Ref<Tensor>& src,
             const Ref<Tensor>& weights,
             const Ref<Tensor>& bias,
             const Ref<Tensor>& dst)
      : MklNode(convolution_forward(desc),
                { { DNNL_ARG_SRC, src->mem },
                  { DNNL_ARG_WEIGHTS, weights->mem },
                  { DNNL_ARG_BIAS, bias->mem },
                  { DNNL_ARG_DST, dst->mem } }),
                src(src), weights(weights), bias(bias), dst(dst)
    {}

    Ref<Tensor> getDst() const override { return dst; }
  };

  // Pooling node
  class PoolNode : public MklNode
  {
  private:
    Ref<Tensor> src;
    Ref<Tensor> dst;

  public:
    PoolNode(const pooling_forward::primitive_desc& desc,
             const Ref<Tensor>& src,
             const Ref<Tensor>& dst)
      : MklNode(pooling_forward(desc),
                { { DNNL_ARG_SRC, src->mem },
                  { DNNL_ARG_DST, dst->mem } }),
                src(src), dst(dst)
    {}

    Ref<Tensor> getDst() const override { return dst; }
  };

  // Reorder node
  class ReorderNode : public MklNode
  {
  private:
    Ref<Tensor> src;
    Ref<Tensor> dst;

  public:
    ReorderNode(const Ref<Tensor>& src,
                const Ref<Tensor>& dst)
      : MklNode(reorder(reorder::primitive_desc(src->mem, dst->mem)),
                { { DNNL_ARG_SRC, src->mem },
                  { DNNL_ARG_DST, dst->mem } }),
                src(src), dst(dst)
    {}

    Ref<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
