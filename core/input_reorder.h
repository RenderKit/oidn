// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"
#include "image.h"
#include "color.h"
#include "reorder.h"

namespace oidn {

  // Input reorder node
  class InputReorderNode : public Node
  {
  protected:
    std::shared_ptr<Image> color;
    std::shared_ptr<Image> albedo;
    std::shared_ptr<Image> normal;
    std::shared_ptr<Tensor> dst;
    std::shared_ptr<TransferFunction> transferFunc;
    ReorderTile tile;
    bool hdr;
    bool snorm;

  public:
    InputReorderNode(const Ref<Device>& device,
                     const std::shared_ptr<Tensor>& dst,
                     const std::shared_ptr<TransferFunction>& transferFunc,
                     bool hdr,
                     bool snorm);

    void setSrc(const std::shared_ptr<Image>& color, const std::shared_ptr<Image>& albedo, const std::shared_ptr<Image>& normal);
    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W);

    std::shared_ptr<Tensor> getDst() const override { return dst; }

  protected:
    int getWidth() const
    {
      return color ? color->width : (albedo ? albedo->width : normal->width);
    }

    int getHeight() const
    {
      return color ? color->height : (albedo ? albedo->height : normal->height);
    }
  };

  class CPUInputReorderNode : public InputReorderNode
  {
  public:
    CPUInputReorderNode(const Ref<Device>& device,
                        const std::shared_ptr<Tensor>& dst,
                        const std::shared_ptr<TransferFunction>& transferFunc,
                        bool hdr,
                        bool snorm);

    void execute() override;
  };

#if defined(OIDN_DEVICE_GPU)

  class SYCLDevice;

  class SYCLInputReorderNode : public InputReorderNode
  {
  public:
    SYCLInputReorderNode(const Ref<SYCLDevice>& device,
                         const std::shared_ptr<Tensor>& dst,
                         const std::shared_ptr<TransferFunction>& transferFunc,
                         bool hdr,
                         bool snorm);

    void execute() override;
  };

#endif

} // namespace oidn
