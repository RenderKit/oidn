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
    TensorDims srcDims;
    int alignment;
    std::shared_ptr<TransferFunction> transferFunc;
    bool hdr;
    bool snorm;
  };

  class InputProcess : public Op, protected InputProcessDesc
  {
  public:
    InputProcess(const Ref<Device>& device, const InputProcessDesc& desc);
    
    TensorDesc getDstDesc() const;
    void setSrc(const std::shared_ptr<Image>& color, const std::shared_ptr<Image>& albedo, const std::shared_ptr<Image>& normal);
    void setDst(const std::shared_ptr<Tensor>& dst);
    std::shared_ptr<Tensor> getDst() const { return dst; }
    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W);

  protected:
    Image* getInput()
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

} // namespace oidn
