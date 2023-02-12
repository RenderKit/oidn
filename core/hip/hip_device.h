// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../device.h"

OIDN_NAMESPACE_BEGIN

#if defined(OIDN_COMPILE_HIP)
  void checkError(hipError_t error);
#endif

  class HIPEngine;

  class HIPDevice final : public Device
  {
    friend class HIPEngine;

  public:
    static bool isSupported();

    explicit HIPDevice(hipStream_t stream = nullptr);
    
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
    hipStream_t stream = nullptr;
    int maxWorkGroupSize = 0;
  };

OIDN_NAMESPACE_END
