// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/device.h"
#if defined(OIDN_DEVICE_CUDA_API_DRIVER)
  #include "curtn.h"
#endif
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
    static bool isSupported(const cudaDeviceProp& prop);
    static bool isSupported(int deviceID);

    CUDADevice(int deviceID, cudaStream_t stream);
    explicit CUDADevice(const Ref<CUDAPhysicalDevice>& physicalDevice);
    ~CUDADevice();

    void enter() override;
    void leave() override;

    DeviceType getType() const override { return DeviceType::CUDA; }

    Storage getPtrStorage(const void* ptr) override;

    void wait() override;

  private:
    void init() override;

    int deviceID = 0;
  #if defined(OIDN_DEVICE_CUDA_API_DRIVER)
    CUdevice deviceHandle = -1;
    CUcontext context = nullptr;
  #else
    int prevDeviceID = -1;
  #endif
    cudaStream_t stream = nullptr;

    int maxWorkGroupSize = 0;
    int subgroupSize = 0;
    int smArch = 0; // compute capability
  };

OIDN_NAMESPACE_END
