// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "memory.h"
#include <vector>

namespace oidn {

  class Node
  {
  public:
    virtual ~Node() = default;

    virtual void execute(stream& sm) = 0;

    virtual std::shared_ptr<memory> getDst() const { return nullptr; }

    virtual size_t getScratchpadSize() const { return 0; }
    virtual void setScratchpad(const std::shared_ptr<memory>& mem) {}

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
    std::shared_ptr<memory> scratchpad;

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

    void setScratchpad(const std::shared_ptr<memory>& mem) override
    {
      scratchpad = mem;
      args.insert(std::make_pair(DNNL_ARG_SCRATCHPAD, *scratchpad));
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
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> bias;
    std::shared_ptr<memory> dst;

  public:
    ConvNode(const convolution_forward::primitive_desc& desc,
             const std::shared_ptr<memory>& src,
             const std::shared_ptr<memory>& weights,
             const std::shared_ptr<memory>& bias,
             const std::shared_ptr<memory>& dst)
      : MklNode(convolution_forward(desc),
                { { DNNL_ARG_SRC, *src },
                  { DNNL_ARG_WEIGHTS, *weights },
                  { DNNL_ARG_BIAS, *bias },
                  { DNNL_ARG_DST, *dst } }),
                src(src), weights(weights), bias(bias), dst(dst)
    {}

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

  // Pooling node
  class PoolNode : public MklNode
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    PoolNode(const pooling_forward::primitive_desc& desc,
             const std::shared_ptr<memory>& src,
             const std::shared_ptr<memory>& dst)
      : MklNode(pooling_forward(desc),
                { { DNNL_ARG_SRC, *src },
                  { DNNL_ARG_DST, *dst } }),
                src(src), dst(dst)
    {}

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

  // Resampling node
  class ResampleNode : public MklNode
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    ResampleNode(const resampling_forward::primitive_desc& desc,
                 const std::shared_ptr<memory>& src,
                 const std::shared_ptr<memory>& dst)
      : MklNode(resampling_forward(desc),
                { { DNNL_ARG_SRC, *src },
                  { DNNL_ARG_DST, *dst } }),
                src(src), dst(dst)
    {}

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

  // Reorder node
  class ReorderNode : public MklNode
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    ReorderNode(const std::shared_ptr<memory>& src,
                const std::shared_ptr<memory>& dst)
      : MklNode(reorder(reorder::primitive_desc(*src, *dst)),
                { { DNNL_ARG_SRC, *src },
                  { DNNL_ARG_DST, *dst } }),
                src(src), dst(dst)
    {}

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

} // namespace oidn
