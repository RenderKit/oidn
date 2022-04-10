// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../image.h"
#include "../image_copy_kernel.h"

namespace oidn {

  namespace
  {
    template<typename ImageT, typename DeviceT>
    void xpuImageCopyKernel(const Ref<DeviceT>& device,
                            const Image& src,
                            const Image& dst)
    {
      ImageCopyKernel<ImageT> kernel;
      kernel.src = src;
      kernel.dst = dst;

      device->parallelFor(dst.getH(), dst.getW(), kernel);
    }
  }

  template<typename DeviceT>
  void xpuImageCopy(const Ref<DeviceT>& device,
                    const Image& src,
                    const Image& dst)
  {
    assert(dst.getH() >= src.getH());
    assert(dst.getW() >= src.getW());

    switch (src.getDataType())
    {
    case DataType::Float32:
      xpuImageCopyKernel<float>(device, src, dst);
      break;
    case DataType::Float16:
      xpuImageCopyKernel<half>(device, src, dst);
      break;
    default:
      assert(0);
    }
  }

} // namespace oidn
