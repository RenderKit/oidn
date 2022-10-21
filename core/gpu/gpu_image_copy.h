// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../image_copy.h"

namespace oidn {

  template<typename T>
  struct GPUImageCopyKernel
  {
    ImageAccessor<T> src;
    ImageAccessor<T> dst;

    OIDN_DEVICE_INLINE void operator ()(const WorkItem<2>& it) const
    {
      const int h = it.getId<0>();
      const int w = it.getId<1>();
      const vec3<T> value = src.get3(h, w);
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

      switch (src->getDataType())
      {
      case DataType::Float32: runImpl<float>(); break;
      case DataType::Float16: runImpl<half>();  break;
      default:                assert(0);
      }
    }

  private:
    template<typename ImageDataT>
    void runImpl()
    {
      GPUImageCopyKernel<ImageDataT> kernel;
      kernel.src = *src;
      kernel.dst = *dst;

      engine->submitKernel(WorkDim<2>(dst->getH(), dst->getW()), kernel);
    }

    Ref<EngineT> engine;
  };

} // namespace oidn
