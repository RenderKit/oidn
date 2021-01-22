// Copyright 2009-2021 Intel Corporation
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

#if defined(OIDN_DNNL)

  // DNNL node base class
  class DNNLNode : public Node
  {
  protected:
    dnnl::primitive prim;
    std::unordered_map<int, dnnl::memory> args;
    Ref<Tensor> scratchpad;

  public:
    DNNLNode(const Ref<Device>& device) : Node(device) {}

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
  };

#elif defined(OIDN_BNNS)

  // BNNS node base class
  class BNNSNode : public Node
  {
  protected:
    BNNSFilter  filter = nullptr;
    const void* inPtr  = nullptr;
    void*       outPtr = nullptr;

  public:
    BNNSNode(const Ref<Device>& device) : Node(device) {}

    ~BNNSNode()
    {
      if (filter)
        BNNSFilterDestroy(filter);
    }

    void execute() override
    {
      BNNSFilterApply(filter, inPtr, outPtr);
    }
  };

#endif

} // namespace oidn
