// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/device.h"
#include "metal_common.h"

OIDN_NAMESPACE_BEGIN

  class MetalEngine;

  class MetalPhysicalDevice : public PhysicalDevice
  {
  public:
    int deviceID;
    
    MetalPhysicalDevice(int deviceID, std::string name, int score);
  };

  class MetalDevice final : public Device
  {
    friend class MetalEngine;
    
  public:
    static std::vector<Ref<PhysicalDevice>> getPhysicalDevices();
    
    MetalDevice(int deviceID = 0);
    MetalDevice(const Ref<MetalPhysicalDevice>& physicalDevice);
    ~MetalDevice();
    
    DeviceType getType() const override { return DeviceType::Metal; }
    
    Engine* getEngine(int i) const override
    {
      assert(i == 0);
      return (Engine*)engine.get();
    }
    
    int getNumEngines() const override { return 1; }
    
    Storage getPointerStorage(const void* ptr) override;
    
    int getInt(const std::string& name) override;
    void setInt(const std::string& name, int value) override;
    
    bool isMemoryUsageLimitSupported() const override { return false; }
    
    void wait() override;
    
    MTLDevice_t getMetalDevice() {
      return device;
    }
    
  protected:
    void init() override;
    
  private:
    Ref<MetalEngine> engine;
    
    int deviceID = 0;
    MTLDevice_t device;
  };

OIDN_NAMESPACE_END
