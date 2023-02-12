// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../device.h"
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
    AVX512_CORE,
    NEON
  };

  class CPUDevice final : public Device
  { 
    friend class CPUEngine;
    friend class DNNLEngine;

  public:
    static bool isSupported();
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

    Storage getPointerStorage(const void* ptr) override;

    int get1i(const std::string& name) override;
    void set1i(const std::string& name, int value) override;

    void wait() override;

  protected:
    void init() override;
    void initTasking();

  private:
    Ref<CPUEngine> engine;
    CPUArch arch;

    // Tasking
    std::shared_ptr<tbb::task_arena> arena;
    std::shared_ptr<PinningObserver> observer;
    std::shared_ptr<ThreadAffinity> affinity;

    int numThreads = 0; // autodetect by default
    bool setAffinity = true;
  };

OIDN_NAMESPACE_END
