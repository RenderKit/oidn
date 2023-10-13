// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/device.h"
#include <cuda_runtime.h>

OIDN_NAMESPACE_BEGIN

  void checkError(cudaError_t error);

  class CUDAEngine;

  class CUDAPhysicalDevice : public PhysicalDevice
  {
  public:
    int deviceID;

    CUDAPhysicalDevice(int deviceID, const cudaDeviceProp& prop, int score);
  };

  class CUDADevice final : public Device
  {
    friend class CUDAEngine;

  public:
    static std::vector<Ref<PhysicalDevice>> getPhysicalDevices();

    CUDADevice(int deviceID = -1, cudaStream_t stream = nullptr);
    explicit CUDADevice(const Ref<CUDAPhysicalDevice>& physicalDevice);
    ~CUDADevice();

    void enter() override;
    void leave() override;

    DeviceType getType() const override { return DeviceType::CUDA; }

    Engine* getEngine(int i) const override
    {
      assert(i == 0);
      return (Engine*)engine.get();
    }

    int getNumEngines() const override { return 1; }

    Storage getPtrStorage(const void* ptr) override;

    void wait() override;

  private:
    void init() override;

    // Supported compute capabilities
    static constexpr int minSMArch = 70;
    static constexpr int maxSMArch = 99;

    Ref<CUDAEngine> engine;

    int deviceID = 0;
    int prevDeviceID = -1;
    cudaStream_t stream = nullptr;

    int maxWorkGroupSize = 0;
    int subgroupSize = 0;
    int smArch = 0; // compute capability
  };

OIDN_NAMESPACE_END
