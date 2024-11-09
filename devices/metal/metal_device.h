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
    static bool isSupported(id<MTLDevice> device);

    explicit MetalDevice(const Ref<MetalPhysicalDevice>& physicalDevice);
    explicit MetalDevice(id<MTLCommandQueue> commandQueue);
    ~MetalDevice();

    DeviceType getType() const override { return DeviceType::Metal; }
    id<MTLDevice> getMTLDevice() const { return device; }

    bool needWeightAndBiasOnDevice() const override { return false; } // due to MPSGraph
    Storage getPtrStorage(const void* ptr) override;

    void execute(std::function<void()>&& f, SyncMode sync) override;
    void flush() override;
    void wait() override;

  protected:
    void init() override;

  private:
    id<MTLDevice> device;
    id<MTLCommandQueue> userCommandQueue = nil; // user-provided command queue (optional)
  };

OIDN_NAMESPACE_END
