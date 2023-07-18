// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/device.h"
#include "metal_common.h"

OIDN_NAMESPACE_BEGIN

  class MetalEngine;

  class MetalPhysicalDevice : public PhysicalDevice
  {
  public:
    id<MTLDevice> device;

    MetalPhysicalDevice(id<MTLDevice> device, int score);
  };

  class MetalDevice final : public Device
  {
    friend class MetalEngine;

  public:
    static std::vector<Ref<PhysicalDevice>> getPhysicalDevices();

    MetalDevice();
    MetalDevice(const Ref<MetalPhysicalDevice>& physicalDevice);
    ~MetalDevice();

    DeviceType getType() const override { return DeviceType::Metal; }

    Engine* getEngine(int i) const override
    {
      assert(i == 0);
      return (Engine*)engine.get();
    }

    int getNumEngines() const override { return 1; }

    id<MTLDevice> getMTLDevice() const { return device; }

    Storage getPtrStorage(const void* ptr) override;
    bool isMemoryUsageLimitSupported() const override { return false; }

    void wait() override;

  protected:
    void init() override;

  private:
    Ref<MetalEngine> engine;

    int deviceID = 0;
    id<MTLDevice> device = nil;
  };

OIDN_NAMESPACE_END
