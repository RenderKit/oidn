// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/device.h"
#include <cuda_runtime.h>

OIDN_NAMESPACE_BEGIN

  void checkError(cudaError_t error);

  class CUDAEngine;

  class CUDADevice final : public Device
  {
    friend class CUDAEngine;

  public:
    static bool isSupported();

    explicit CUDADevice(cudaStream_t stream = nullptr);
    
    DeviceType getType() const override { return DeviceType::CUDA; }

    Engine* getEngine(int i) const override
    {
      assert(i == 0);
      return (Engine*)engine.get();
    }
    
    int getNumEngines() const override { return 1; }

    Storage getPointerStorage(const void* ptr) override;

    void wait() override;

  private:
    void init() override;

    // Supported compute capabilities
    static constexpr int minSMArch = 70;
    static constexpr int maxSMArch = 99;

    Ref<CUDAEngine> engine;
    cudaStream_t stream = nullptr;

    int maxWorkGroupSize = 0;
    int smArch = 0; // compute capability
  };

OIDN_NAMESPACE_END
