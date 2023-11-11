// Copyright 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "unet_filter.h"

OIDN_NAMESPACE_BEGIN

  // RTLightmap: Ray traced lightmap denoiser
  class RTLightmapFilter final : public UNetFilter
  {
  public:
    explicit RTLightmapFilter(const Ref<Device>& device);

    void setImage(const std::string& name, const Ref<Image>& image) override;
    void unsetImage(const std::string& name) override;
    void setInt(const std::string& name, int value) override;
    int getInt(const std::string& name) override;

  protected:
    std::shared_ptr<TransferFunction> newTransferFunc() override;
  };

OIDN_NAMESPACE_END
