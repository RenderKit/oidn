// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/device.h"

OIDN_NAMESPACE_BEGIN

#if defined(OIDN_COMPILE_HIP)
  void checkError(hipError_t error);
#endif

  class HIPEngine;

  // GPU matrix architecture
  enum class HIPArch
  {
    Unknown,
    DL,      // RDNA2
    XDL,     // CDNA1, CDNA2
    WMMA,    // RDNA3
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
    static HIPArch getArch(const std::string& archStr);

    HIPDevice(int deviceID = -1, hipStream_t stream = nullptr);
    explicit HIPDevice(const Ref<HIPPhysicalDevice>& physicalDevice);
    ~HIPDevice();

    void begin() override;
    void end() override;
    
    DeviceType getType() const override { return DeviceType::HIP; }

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

    Ref<HIPEngine> engine;

    int deviceID = 0;
    int prevDeviceID = -1;
    hipStream_t stream = nullptr;

    HIPArch arch = HIPArch::Unknown;
    int maxWorkGroupSize = 0;
  };

OIDN_NAMESPACE_END
