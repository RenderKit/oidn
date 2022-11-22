// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../device.h"
#include <cuda_runtime.h>

namespace oidn {

  void checkError(cudaError_t error);

  class CUDAEngine;

  class CUDADevice final : public Device
  {
    friend class CUDAEngine;

  public:
    static bool isSupported();

    explicit CUDADevice(cudaStream_t stream = nullptr);

    Engine* getEngine(int i) const override
    {
      assert(i == 0);
      return (Engine*)engine.get();
    }
    
    int getNumEngines() const override { return 1; }

    void wait() override;

  private:
    void init() override;

    static constexpr int minComputeCapability = 70;
    static constexpr int maxComputeCapability = 87;

    Ref<CUDAEngine> engine;
    cudaStream_t stream = nullptr;

    int maxWorkGroupSize = 0;
    int computeCapability = 0;
  };

} // namespace oidn
