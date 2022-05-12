// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../image_copy.h"
#include "../device.h"

namespace oidn {

  class CPUImageCopy final : public ImageCopy
  {
  public:
    explicit CPUImageCopy(const Ref<Device>& device);
    void run() override;

  private:
    Ref<Device> device;
  };

} // namespace oidn
