// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../image_copy.h"
#include "cpu_op.h"

namespace oidn {

  class CPUImageCopy final : public CPUOp, public ImageCopy
  {
  public:
    explicit CPUImageCopy(const Ref<CPUDevice>& device);
    void run() override;
  };

} // namespace oidn
