// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor.h"
#include <vector>

namespace oidn {

  class Node : public RefCount
  {
  private:
    Ref<Device> device;

  public:
    Node(const Ref<Device>& device) : device(device) {}
    virtual ~Node() = default;

    virtual void execute() = 0;

    virtual Ref<Tensor> getDst() const { return nullptr; }

    virtual size_t getScratchpadSize() const { return 0; }
    virtual void setScratchpad(const Ref<Tensor>& scratchpad) {}

    virtual void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W)
    {
      assert(0); // not supported
    }

    __forceinline Device* getDevice() { return device.get(); }
  };

  // Node wrapping a DNNL primitive
  class DNNLNode : public Node
  {
  private:
    dnnl::primitive prim;
    std::unordered_map<int, dnnl::memory> args;
    Ref<Tensor> scratchpad;

  public:
    DNNLNode(const Ref<Device>& device)
      : Node(device)
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

    void execute() override
    {
      prim.execute(getDevice()->getDNNLStream(), args);
    }

  protected:
    void setPrimitive(const dnnl::primitive prim)
    {
      this->prim = prim;
    }

    void setArgs(const std::unordered_map<int, dnnl::memory>& args)
    {
      this->args = args;
    }
  };

  // Reorder node
  class ReorderNode : public DNNLNode
  {
  private:
    Ref<Tensor> src;
    Ref<Tensor> dst;

  public:
    ReorderNode(const Ref<Device>& device,
                const Ref<Tensor>& src,
                const Ref<Tensor>& dst)
      : DNNLNode(device),
        src(src), dst(dst)
    {
      setPrimitive(dnnl::reorder(dnnl::reorder::primitive_desc(src->mem, dst->mem)));
      setArgs({{DNNL_ARG_SRC, src->mem},
               {DNNL_ARG_DST, dst->mem}});
    }

    Ref<Tensor> getDst() const override { return dst; }
  };

  // Convolution node
  class ConvNode : public DNNLNode
  {
  private:
    Ref<Tensor> src;
    Ref<Tensor> weights;
    Ref<Tensor> bias;
    Ref<Tensor> dst;

  public:
    ConvNode(const Ref<Device>& device,
             const Ref<Tensor>& src,
             const Ref<Tensor>& weights,
             const Ref<Tensor>& bias,
             const Ref<Tensor>& dst,
             bool relu)
      : DNNLNode(device),
        src(src), weights(weights), bias(bias), dst(dst)
    {
      const dnnl::memory::dims strides = {1, 1};
      const dnnl::memory::dims padding = {1, 1};

      // Let the convolution primitive choose the weights format
      auto weightsDesc = dnnl::memory::desc({ weights->dims },
                                            dnnl::memory::data_type::f32,
                                            dnnl::memory::format_tag::any);

      auto convDesc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
        src->mem.get_desc(),
        weightsDesc,
        bias->mem.get_desc(),
        dst->mem.get_desc(),
        strides, padding, padding);

      // Incorporate relu
      dnnl::primitive_attr convAttr;
      if (relu)
      {
        dnnl::post_ops ops;
        ops.append_eltwise(
          1.f,   // scale
          dnnl::algorithm::eltwise_relu,
          0.f,   // alpha
          0.f    // beta
        );
        convAttr.set_post_ops(ops);
      }
      convAttr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

      auto convPrimDesc = dnnl::convolution_forward::primitive_desc(convDesc, convAttr, device->getDNNLEngine());

      // Reorder the weights to the final format, if necessary
      if (convPrimDesc.weights_desc() != weights->mem.get_desc())
      {
        this->weights = makeRef<Tensor>(device, convPrimDesc.weights_desc());
        ReorderNode(device, weights, this->weights).execute();
      }

      setPrimitive(dnnl::convolution_forward(convPrimDesc));
      setArgs({{DNNL_ARG_SRC,     src->mem},
               {DNNL_ARG_WEIGHTS, this->weights->mem},
               {DNNL_ARG_BIAS,    this->bias->mem},
               {DNNL_ARG_DST,     dst->mem}});
    }

    Ref<Tensor> getDst() const override { return dst; }
  };

  // Pooling node
  class PoolNode : public DNNLNode
  {
  private:
    Ref<Tensor> src;
    Ref<Tensor> dst;

  public:
    PoolNode(const Ref<Device>& device,
             const Ref<Tensor>& src,
             const Ref<Tensor>& dst)
      : DNNLNode(device),
        src(src), dst(dst)
    {
      const dnnl::memory::dims kernel  = {2, 2};
      const dnnl::memory::dims strides = {2, 2};
      const dnnl::memory::dims padding = {0, 0};

      auto poolDesc = dnnl::pooling_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_max,
        src->mem.get_desc(),
        dst->mem.get_desc(),
        strides, kernel, padding, padding);

      dnnl::primitive_attr poolAttr;
      poolAttr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

      auto poolPrimDesc = dnnl::pooling_forward::primitive_desc(poolDesc, poolAttr, device->getDNNLEngine());

      setPrimitive(dnnl::pooling_forward(poolPrimDesc));
      setArgs({{DNNL_ARG_SRC, src->mem},
               {DNNL_ARG_DST, dst->mem}});
    }

    Ref<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
