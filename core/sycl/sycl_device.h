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
    static bool isDeviceSupported(const sycl::device& syclDevice);
    static SYCLArch getDeviceArch(const sycl::device& syclDevice);

    SYCLDevice(const std::vector<sycl::queue>& syclQueues = {});
    
    Engine* getEngine(int i) const override { return (Engine*)engines[i].get(); }
    int getNumEngines() const override { return int(engines.size()); }
    
    int get1i(const std::string& name) override;
    void set1i(const std::string& name, int value) override;

    Storage getPointerStorage(const void* ptr) override;

    void submitBarrier() override;
    void wait() override;
    
    // Manually sets the dependent events for the next command on all engines
    void setDepEvents(const std::vector<sycl::event>& depEvents);
    
    // Gets the list of events corresponding to the completion of all commands
    std::vector<sycl::event> getDoneEvents();

    SYCLArch getArch() const { return arch; }

  private:
    void init() override;

    sycl::context syclContext;
    std::vector<sycl::queue> syclQueues; // used only for initialization
    std::vector<Ref<SYCLEngine>> engines;
    SYCLArch arch;
    
    
    int numSubdevices = 0; // autodetect by default
  };

} // namespace oidn
