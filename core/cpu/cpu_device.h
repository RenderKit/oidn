// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../device.h"

#if defined(OIDN_DNNL)
  #include "../dnnl/dnnl_device.h"
#elif defined(OIDN_BNNS)
  #include "../bnns/bnns_device.h"
#endif
#include "tasking.h"

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
    CPUDevice();
    ~CPUDevice();

    int get1i(const std::string& name) override;
    void set1i(const std::string& name, int value) override;

    // Ops
    std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) override;
    std::shared_ptr<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;
    std::shared_ptr<ImageCopy> newImageCopy() override;

    // Runs a parallel host task in the thread arena (if it exists)
    void runHostTask(std::function<void()>&& f) override;

    // Enqueues a host function
    void runHostFuncAsync(std::function<void()>&& f) override;

  protected:
    void init() override;
    void initTasking();

  private:
    // Tasking
    std::shared_ptr<tbb::task_arena> arena;
    std::shared_ptr<PinningObserver> observer;
    std::shared_ptr<ThreadAffinity> affinity;
    int numThreads = 0; // autodetect by default
    bool setAffinity = true;
  };

} // namespace oidn
