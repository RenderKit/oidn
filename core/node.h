// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor.h"

namespace oidn {

  // Node base class
  class Node
  {
  protected:
    Ref<Device> device;

  public:
    Node(const Ref<Device>& device) : device(device) {}
    virtual ~Node() = default;

    virtual void execute() = 0;

    virtual std::shared_ptr<Tensor> getDst() const { return nullptr; }

    virtual size_t getScratchSize() const { return 0; }
    virtual void setScratch(const std::shared_ptr<Tensor>& scratch) {}

    __forceinline Device* getDevice() { return device.get(); }
  };

#if defined(OIDN_DNNL)

  // DNNL node base class
  class DNNLNode : public Node
  {
  protected:
    dnnl::primitive prim;
    std::unordered_map<int, dnnl::memory> args;
    std::shared_ptr<Tensor> scratch;

  public:
    DNNLNode(const Ref<Device>& device) : Node(device) {}

    size_t getScratchSize() const override
    {
      const auto primDesc = prim.get_primitive_desc();
      const dnnl_memory_desc_t* scratchpadDesc = dnnl_primitive_desc_query_md(primDesc, dnnl_query_scratchpad_md, 0);
      if (scratchpadDesc == nullptr)
        return 0;
      return dnnl_memory_desc_get_size(scratchpadDesc);
    }

    void setScratch(const std::shared_ptr<Tensor>& scratch) override
    {
      this->scratch = scratch;
      args.insert(std::make_pair(DNNL_ARG_SCRATCHPAD, scratch->mem));
    }

    void execute() override
    {
      prim.execute(device->getDNNLStream(), args);
    }
  };

#elif defined(OIDN_BNNS)

  // BNNS node base class
  class BNNSNode : public Node
  {
  protected:
    BNNSFilter filter = nullptr;

  public:
    BNNSNode(const Ref<Device>& device) : Node(device) {}

    ~BNNSNode()
    {
      if (filter)
        BNNSFilterDestroy(filter);
    }
  };

#endif

} // namespace oidn
