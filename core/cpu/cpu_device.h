// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../device.h"

#if defined(OIDN_DNNL)
  #include "../dnnl/dnnl_device.h"
#elif defined(OIDN_BNNS)
  #include "../bnns/bnns_device.h"
#endif

namespace oidn {

  class CPUDevice
  #if defined(OIDN_DNNL)
    : public DNNLDevice
  #elif defined(OIDN_BNNS)
    : public BNNSDevice
  #else
    : public Device
  #endif
  { 
  private:
    // Tasking
    std::shared_ptr<PinningObserver> observer;
    std::shared_ptr<ThreadAffinity> affinity;

  public:
    ~CPUDevice();

    Ref<Buffer> newBuffer(size_t byteSize, Buffer::Kind kind) override;
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize) override;

    // Nodes
    std::shared_ptr<UpsampleNode> newUpsampleNode(const UpsampleDesc& desc) override;
    std::shared_ptr<InputReorderNode> newInputReorderNode(const InputReorderDesc& desc) override;
    std::shared_ptr<OutputReorderNode> newOutputReorderNode(const OutputReorderDesc& desc) override;

    // Kernels
    void imageCopy(const Image& src, const Image& dst) override;

    // Executes a kernel on the device
    template <typename T0, typename T1, typename F>
    OIDN_INLINE void executeKernel(const T0& D0, const T1& D1, const F& f)
    {
      parallel_nd(D0, D1, f);
    }

  protected:
    void init() override;
    void printInfo() override;
    void initTasking();
  };

} // namespace oidn
