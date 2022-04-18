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

  template<typename OpType>
  class GPUImageCopy final : public OpType, public ImageCopy
  {
  public:
    explicit GPUImageCopy(const Ref<typename OpType::DeviceType>& device) : OpType(device) {}

    void run() override
    {
      assert(dst->getH() >= src->getH());
      assert(dst->getW() >= src->getW());
      assert(dst->getDataType() == src->getDataType());

      switch (src->getDataType())
      {
      case DataType::Float32: runImpl<float>(); break;
      case DataType::Float16: runImpl<half>();  break;
      default:                assert(0);
      }
    }

  private:
    template<typename ImageDataType>
    void runImpl()
    {
      GPUImageCopyKernel<ImageDataType> kernel;
      kernel.src = *src;
      kernel.dst = *dst;

      this->device->runKernelAsync({dst->getH(), dst->getW()}, kernel);
    }
  };

} // namespace oidn
