// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"

namespace oidn {

  class CPUDevice : public Device
  { 
  public:
    void commit() override;

    Ref<Buffer> newBuffer(size_t byteSize) override;
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize) override;
  };

} // namespace oidn
