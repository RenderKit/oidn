// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/image_copy.h"

OIDN_NAMESPACE_BEGIN

  struct GPUImageCopyKernel
  {
    ImageAccessor src;
    ImageAccessor dst;

    OIDN_DEVICE_INLINE void operator ()(const WorkItem<2>& it) const
    {
      const int h = it.getId<0>();
      const int w = it.getId<1>();
      const vec3f value = src.get3(h, w);
      dst.set3(h, w, value);
    }
  };

  template<typename EngineT>
  class GPUImageCopy final : public ImageCopy
  {
  public:
    explicit GPUImageCopy(const Ref<EngineT>& engine)
      : engine(engine) {}

    void submit() override
    {
      if (!src || !dst)
        throw std::logic_error("image copy source/destination not set");
      if (dst->getH() < src->getH() || dst->getW() < src->getW())
        throw std::out_of_range("image copy destination smaller than the source");
      if (dst->getDataType() != src->getDataType())
        throw std::invalid_argument("image copy source and destination have different data types");

      GPUImageCopyKernel kernel;
      kernel.src = *src;
      kernel.dst = *dst;

      engine->submitKernel(WorkDim<2>(dst->getH(), dst->getW()), kernel);
    }

  private:
    Ref<EngineT> engine;
  };

OIDN_NAMESPACE_END
