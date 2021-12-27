// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "image.h"
#include "image_copy_kernel.h"

namespace oidn {

  namespace
  {
    template<typename ImageDT, typename DeviceT>
    void xpuImageCopyKernel(const Ref<DeviceT>& device,
                            const Image& src,
                            const Image& dst)
    {
      ImageCopy<ImageDT> kernel;
      kernel.src = src;
      kernel.dst = dst;

      device->executeKernel(dst.height, dst.width, kernel);
    }
  }

  template<typename DeviceT>
  void xpuImageCopy(const Ref<DeviceT>& device,
                    const Image& src,
                    const Image& dst)
  {
    assert(dst.height >= src.height);
    assert(dst.width  >= src.width);

    switch (getDataType(src.format))
    {
    case DataType::Float32: xpuImageCopyKernel<float>(device, src, dst); break;
    case DataType::Float16: xpuImageCopyKernel<half>(device, src, dst);  break;
    default:                assert(0);
    }
  }

} // namespace oidn
