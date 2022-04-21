// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "input_process.h"

namespace oidn {

  InputProcess::InputProcess(const Ref<Device>& device, const InputProcessDesc& desc)
    : InputProcessDesc(desc)
  {
    assert(srcDims.size() == 3);

    TensorDims dstDims {
      round_up(srcDims[0], device->getTensorBlockSize()), // round up C
      round_up(srcDims[1], int64_t(alignment)), // round up H
      round_up(srcDims[2], int64_t(alignment))  // round up W
    };
    dstDesc = {dstDims, device->getTensorLayout(), device->getTensorDataType()};

    setTile(0, 0, 0, 0, 0, 0);
  }

  TensorDesc InputProcess::getDstDesc() const
  {
    return dstDesc;
  }

  void InputProcess::setSrc(const std::shared_ptr<Image>& color, const std::shared_ptr<Image>& albedo, const std::shared_ptr<Image>& normal)
  {
    // FIXME: add checks
    this->color  = color;
    this->albedo = albedo;
    this->normal = normal;
  }

  void InputProcess::setDst(const std::shared_ptr<Tensor>& dst)
  {
    assert(dst->getDesc() == getDstDesc());
    this->dst = dst;
  }

  void InputProcess::setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W)
  {
    tile.hSrcBegin = hSrc;
    tile.wSrcBegin = wSrc;
    tile.hDstBegin = hDst;
    tile.wDstBegin = wDst;
    tile.H = H;
    tile.W = W;
  }

} // namespace oidn