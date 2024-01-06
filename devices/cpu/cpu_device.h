// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/device.h"
#if defined(OIDN_DNNL)
  #include "mkl-dnn/include/dnnl.hpp"
#endif
#include "tasking.h"

OIDN_NAMESPACE_BEGIN

  class CPUEngine;

  // CPU instruction set
  enum class CPUArch
  {
    Unknown,
    SSE41,
    AVX2,
    AVX512,
    NEON
  };

  class CPUPhysicalDevice final : public PhysicalDevice
  {
  public:
    explicit CPUPhysicalDevice(int score);
  };

  class CPUDevice final : public Device
  {
    friend class CPUEngine;
    friend class DNNLEngine;

  public:
    static std::vector<Ref<PhysicalDevice>> getPhysicalDevices();
    static std::string getName();
    static CPUArch getArch();

    CPUDevice();
    ~CPUDevice();

    DeviceType getType() const override { return DeviceType::CPU; }

    Engine* getEngine(int i) const override
    {
      assert(i == 0);
      return (Engine*)engine.get();
    }

    int getNumEngines() const override { return 1; }

    Storage getPtrStorage(const void* ptr) override;

    int getInt(const std::string& name) override;
    void setInt(const std::string& name, int value) override;

    void wait() override;

  protected:
    void init() override;
    void initTasking();

  private:
    std::unique_ptr<CPUEngine> engine;
    CPUArch arch = CPUArch::Unknown;

    // Tasking
    std::shared_ptr<tbb::task_arena> arena;
    std::shared_ptr<PinningObserver> observer;
    std::shared_ptr<ThreadAffinity> affinity;

    int numThreads = 0; // autodetect by default
    bool setAffinity = true;
  };

OIDN_NAMESPACE_END
