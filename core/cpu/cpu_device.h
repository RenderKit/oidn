// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../device.h"

#if defined(OIDN_DNNL)
  #include "../dnnl/dnnl_device.h"
#elif defined(OIDN_BNNS)
  #include "../bnns/bnns_device.h"
#endif

namespace oidn {

  class CPUDevice final
  #if defined(OIDN_DNNL)
    : public DNNLDevice
  #elif defined(OIDN_BNNS)
    : public BNNSDevice
  #else
    : public Device
  #endif
  { 
  public:
    ~CPUDevice();

    // Ops
    std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) override;
    std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;

    // Kernels
    void imageCopy(const Image& src, const Image& dst) override;

    // Runs a kernel on the device
    template<typename Ty, typename Tx, typename F>
    OIDN_INLINE void runKernel(const Ty& Dy, const Tx& Dx, const F& f)
    {
      parallel_nd(Dy, Dx, f);
    }

  protected:
    void init() override;
    void printInfo() override;
    void initTasking();

  private:
    // Tasking
    std::shared_ptr<PinningObserver> observer;
    std::shared_ptr<ThreadAffinity> affinity;
  };

} // namespace oidn
