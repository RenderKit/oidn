// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor.h"

namespace oidn {

  // Node base class
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

  // DNNL node base class
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

} // namespace oidn
