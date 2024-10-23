// Copyright 2018 Intel Corporation
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

  class OutputProcess : public BaseOp, protected OutputProcessDesc
  {
  public:
    OutputProcess(const OutputProcessDesc& desc);

    TensorDesc getSrcDesc() const { return srcDesc; }
    Ref<Tensor> getSrc() const { return src; }

    void setSrc(const Ref<Tensor>& src);
    void setDst(const Ref<Image>& dst);
    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W);

  protected:
    void check();

    Ref<Tensor> src;
    Ref<Image> dst;
    Tile tile;
  };

OIDN_NAMESPACE_END
