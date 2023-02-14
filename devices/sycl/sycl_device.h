// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/device.h"
#include <level_zero/ze_api.h>

OIDN_NAMESPACE_BEGIN

  class SYCLEngine;

  // GPU architecture
  enum class SYCLArch
  {
    Unknown,
    Gen9,
    XeHPG,
    XeHPC,
  };

  class SYCLDevice : public SYCLDeviceBase
  { 
  public:
    static bool isSupported();
    static SYCLArch getArch(const sycl::device& syclDevice);

    SYCLDevice(const std::vector<sycl::queue>& syclQueues = {});

    DeviceType getType() const override { return DeviceType::SYCL; }
    ze_context_handle_t getZeContext() const { return zeContext; }
    
    Engine* getEngine(int i) const override { return (Engine*)engines[i].get(); }
    int getNumEngines() const override { return int(engines.size()); }
    
    int get1i(const std::string& name) override;
    void set1i(const std::string& name, int value) override;

    Storage getPointerStorage(const void* ptr) override;

    void submitBarrier() override;
    void wait() override;
    
    // Manually sets the dependent events for the next command on all engines
    void setDepEvents(const std::vector<sycl::event>& events);
    void setDepEvents(const sycl::event* events, int numEvents) override;
    
    // Gets the list of events corresponding to the completion of all commands
    std::vector<sycl::event> getDoneEvents();
    void getDoneEvent(sycl::event& event) override;

    SYCLArch getArch() const { return arch; }

  private:
    void init() override;

    sycl::context syclContext;
    ze_context_handle_t zeContext = nullptr; // Level Zero context
    std::vector<sycl::queue> syclQueues;     // used only for initialization
    std::vector<Ref<SYCLEngine>> engines;
    SYCLArch arch;
    
    int numSubdevices = 0; // autodetect by default
  };

OIDN_NAMESPACE_END
