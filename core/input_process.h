// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "image.h"
#include "color.h"
#include "tile.h"

namespace oidn {

  struct InputProcessDesc
  {
    std::shared_ptr<Tensor> dst;
    std::shared_ptr<TransferFunction> transferFunc;
    bool hdr;
    bool snorm;
  };

  class InputProcess : public virtual Op
  {
  protected:
    std::shared_ptr<Image> color;
    std::shared_ptr<Image> albedo;
    std::shared_ptr<Image> normal;
    std::shared_ptr<Tensor> dst;
    std::shared_ptr<TransferFunction> transferFunc;
    Tile tile;
    bool hdr;
    bool snorm;

  public:
    InputProcess(const InputProcessDesc& desc);

    void setSrc(const std::shared_ptr<Image>& color, const std::shared_ptr<Image>& albedo, const std::shared_ptr<Image>& normal);
    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W);

    std::shared_ptr<Tensor> getDst() const override { return dst; }

  protected:
    Image* getInput()
    {
      return color ? color.get() : (albedo ? albedo.get() : normal.get());
    }
  };

} // namespace oidn
