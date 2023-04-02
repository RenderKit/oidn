// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "image.h"
#include "tensor.h"
#include "color.h"
#include "tile.h"

OIDN_NAMESPACE_BEGIN

  struct OutputProcessDesc
  {
    TensorDesc srcDesc;
    std::shared_ptr<TransferFunction> transferFunc;
    bool hdr;
    bool snorm;
  };

  class OutputProcess : public Op, protected OutputProcessDesc
  {
  public:
    OutputProcess(const OutputProcessDesc& desc);
    
    void setSrc(const std::shared_ptr<Tensor>& src);
    void setDst(const std::shared_ptr<Image>& dst);
    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W);

  protected:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Image> dst;
    Tile tile;
  };

OIDN_NAMESPACE_END
