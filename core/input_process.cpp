// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "input_process.h"

namespace oidn {

  InputProcess::InputProcess(const Ref<Device>& device, const InputProcessDesc& desc)
    : InputProcessDesc(desc)
  {
    if (srcDims.size() != 3)
      throw std::invalid_argument("invalid input processing source shape");

    TensorDims dstDims {
      round_up(srcDims[0], device->getTensorBlockC()), // round up C
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

  void InputProcess::setSrc(const std::shared_ptr<Image>& color,
                            const std::shared_ptr<Image>& albedo,
                            const std::shared_ptr<Image>& normal)
  {
    int C = 0;
    if (color)  C += color->getC();
    if (albedo) C += albedo->getC();
    if (normal) C += normal->getC();
    if (C != srcDims[0])
      throw std::invalid_argument("invalid input processing source");

    this->color  = color;
    this->albedo = albedo;
    this->normal = normal;
    updateSrc();
  }

  void InputProcess::setDst(const std::shared_ptr<Tensor>& dst)
  {
    if (!dst || dst->getDesc() != dstDesc)
      throw std::invalid_argument("invalid input processing destination");

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