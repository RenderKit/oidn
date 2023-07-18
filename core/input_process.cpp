// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "input_process.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  InputProcess::InputProcess(const Ref<Engine>& engine, const InputProcessDesc& desc)
    : InputProcessDesc(desc)
  {
    if (srcDims.size() != 3)
      throw std::invalid_argument("invalid input processing source shape");

    TensorDims dstDims {
      srcDims[0],
      round_up(srcDims[1], tileAlignment), // round up H
      round_up(srcDims[2], tileAlignment)  // round up W
    };

    TensorDims dstPaddedDims {
      round_up(srcDims[0], engine->getDevice()->getTensorBlockC()), // round up C
      dstDims[1],
      dstDims[2]
    };

    dstDesc = {dstDims, dstPaddedDims, engine->getDevice()->getTensorLayout(), engine->getDevice()->getTensorDataType()};

    setTile(0, 0, 0, 0, 0, 0);
  }

  void InputProcess::setSrc(const std::shared_ptr<Image>& color,
                            const std::shared_ptr<Image>& albedo,
                            const std::shared_ptr<Image>& normal)
  {
    int C = 0;
    if (color)  C += 3; // always broadcast to 3 channels
    if (albedo) C += 3;
    if (normal) C += 3;
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

  void InputProcess::check()
  {
    if (!getMainSrc() || !dst)
      throw std::logic_error("input processing source/destination not set");
    if (tile.hSrcBegin + tile.H > getMainSrc()->getH() ||
        tile.wSrcBegin + tile.W > getMainSrc()->getW() ||
        tile.hDstBegin + tile.H > dst->getH() ||
        tile.wDstBegin + tile.W > dst->getW())
      throw std::out_of_range("input processing source/destination out of range");
  }

OIDN_NAMESPACE_END