// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/image_copy.h"
#include "metal_engine.h"

OIDN_NAMESPACE_BEGIN

  class MetalImageCopy final : public ImageCopy
  {
  public:
    explicit MetalImageCopy(const Ref<MetalEngine>& engine);
    void submit() override;

  private:
    Ref<MetalEngine> engine;
  };

OIDN_NAMESPACE_END
