// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "image.h"
#include "color.h"
#include "tile.h"

OIDN_NAMESPACE_BEGIN

  struct InputProcessDesc
  {
    TensorDims srcDims;
    int alignment;
    std::shared_ptr<TransferFunction> transferFunc;
    bool hdr;
    bool snorm;
  };

  class InputProcess : public Op, protected InputProcessDesc
  {
  public:
    InputProcess(const Ref<Engine>& engine, const InputProcessDesc& desc);
    
    TensorDesc getDstDesc() const;
    std::shared_ptr<Tensor> getDst() const { return dst; }

    void setSrc(const std::shared_ptr<Image>& color,
                const std::shared_ptr<Image>& albedo,
                const std::shared_ptr<Image>& normal);
    void setDst(const std::shared_ptr<Tensor>& dst);
    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W);

  protected:
    virtual void updateSrc() {}

    Image* getMainSrc()
    {
      return color ? color.get() : (albedo ? albedo.get() : normal.get());
    }

    TensorDesc dstDesc;
    std::shared_ptr<Image> color;
    std::shared_ptr<Image> albedo;
    std::shared_ptr<Image> normal;
    std::shared_ptr<Tensor> dst;
    Tile tile;
  };

OIDN_NAMESPACE_END
