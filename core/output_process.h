// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "image.h"
#include "color.h"
#include "tile.h"

namespace oidn {

  struct OutputProcessDesc
  {
    std::shared_ptr<Tensor> src;
    std::shared_ptr<TransferFunction> transferFunc;
    bool hdr;
    bool snorm;
  };

  class OutputProcess : public virtual Op
  {
  protected:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Image> output;
    std::shared_ptr<TransferFunction> transferFunc;
    Tile tile;
    bool hdr;
    bool snorm;

  public:
    OutputProcess(const OutputProcessDesc& desc);

    void setDst(const std::shared_ptr<Image>& output);
    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W);
  };

} // namespace oidn
