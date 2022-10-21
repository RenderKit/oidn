// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../device.h"

namespace oidn {

  class SYCLEngine;

  // GPU architecture
  enum class SYCLArch
  {
    Gen9,
    XeHPG,
    XeHPC,
  };

  class SYCLDevice : public Device
  { 
  public:
    static bool isSupported();
    static bool isDeviceSupported(const sycl::device& device);
    static SYCLArch getDeviceArch(const sycl::device& device);

    SYCLDevice();
    SYCLDevice(const std::vector<sycl::queue>& queues);
    
    Engine* getEngine(int i) const override { return (Engine*)engines[i].get(); }
    int getNumEngines() const override { return int(engines.size()); }

    void submitBarrier() override;
    void wait() override;
    
    // Manually sets the dependencies for the next command on all engines
    void setDependencies(const std::vector<sycl::event>& depEvents);
    
    // Gets the list of events corresponding to the completion of all commands
    std::vector<sycl::event> getDone();

    SYCLArch getArch() const { return arch; }

  private:
    void init() override;

    std::vector<sycl::queue> queues; // used only for initialization
    std::vector<Ref<SYCLEngine>> engines;
    SYCLArch arch;
  };

} // namespace oidn
