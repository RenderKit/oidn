// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/device.h"

OIDN_NAMESPACE_BEGIN

  void checkError(hipError_t error);

  class HIPEngine;

  // GPU matrix architecture
  enum class HIPArch
  {
    Unknown,
    DL,      // RDNA 2
    WMMA,    // RDNA 3
  };

  class HIPPhysicalDevice : public PhysicalDevice
  {
  public:
    int deviceID;

    HIPPhysicalDevice(int deviceID, const hipDeviceProp_t& prop, int score);
  };

  class HIPDevice final : public Device
  {
    friend class HIPEngine;

  public:
    static std::vector<Ref<PhysicalDevice>> getPhysicalDevices();
    static std::string getName(const hipDeviceProp_t& prop);
    static std::string getArchName(const hipDeviceProp_t& prop);
    static HIPArch getArch(const hipDeviceProp_t& prop);
    static bool isSupported(int deviceID);

    HIPDevice(int deviceID, hipStream_t stream);
    explicit HIPDevice(const Ref<HIPPhysicalDevice>& physicalDevice);
    ~HIPDevice();

    void enter() override;
    void leave() override;

    DeviceType getType() const override { return DeviceType::HIP; }

    Storage getPtrStorage(const void* ptr) override;

    void wait() override;

  private:
    void init() override;

    int deviceID = 0;
    int prevDeviceID = -1;
    hipStream_t stream = nullptr;

    HIPArch arch = HIPArch::Unknown;
    int maxWorkGroupSize = 0;
    int subgroupSize = 0;
  };

OIDN_NAMESPACE_END
