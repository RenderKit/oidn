// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../image.h"

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

  namespace
  {
    template<typename ImageT, typename DeviceT>
    void gpuImageCopyKernel(const Ref<DeviceT>& device,
                            const Image& src,
                            const Image& dst)
    {
      GPUImageCopyKernel<ImageT> kernel;
      kernel.src = src;
      kernel.dst = dst;

      device->runKernel({dst.getH(), dst.getW()}, kernel);
    }
  }

  template<typename DeviceT>
  void gpuImageCopy(const Ref<DeviceT>& device,
                    const Image& src,
                    const Image& dst)
  {
    assert(dst.getH() >= src.getH());
    assert(dst.getW() >= src.getW());

    switch (src.getDataType())
    {
    case DataType::Float32:
      gpuImageCopyKernel<float>(device, src, dst);
      break;
    case DataType::Float16:
      gpuImageCopyKernel<half>(device, src, dst);
      break;
    default:
      assert(0);
    }
  }

} // namespace oidn
