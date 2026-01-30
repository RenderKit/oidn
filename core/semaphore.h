// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"

OIDN_NAMESPACE_BEGIN

  class Semaphore : public RefCount
  {
  public:
    explicit Semaphore(const Ref<Device>& device) : device(device) {}

    Device* getDevice() const { return device.get(); }

  protected:
    Ref<Device> device;
  };

OIDN_NAMESPACE_END