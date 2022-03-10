// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "unet_filter.h"

namespace oidn {

  // RT: Generic ray tracing denoiser
  class RTFilter final : public UNetFilter
  {
  public:
    explicit RTFilter(const Ref<Device>& device);

    void setImage(const std::string& name, const std::shared_ptr<Image>& image) override;
    void removeImage(const std::string& name) override;
    void set1i(const std::string& name, int value) override;
    int get1i(const std::string& name) override;
  
  protected:
    std::shared_ptr<TransferFunction> newTransferFunc() override;
  };

} // namespace oidn
