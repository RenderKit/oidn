// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include "metal_image_copy.h"
#include "metal_engine.h"

OIDN_NAMESPACE_BEGIN

  MetalImageCopy::MetalImageCopy(const Ref<MetalEngine>& engine)
    : engine(engine) {}

  void MetalImageCopy::submit()
  {
    throw std::logic_error("Not implemented");
  }

OIDN_NAMESPACE_END
