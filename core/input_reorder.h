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
                     const std::string& name,
                     const std::shared_ptr<Tensor>& dst,
                     const std::shared_ptr<TransferFunction>& transferFunc,
                     bool hdr,
                     bool snorm);

    void setSrc(const std::shared_ptr<Image>& color, const std::shared_ptr<Image>& albedo, const std::shared_ptr<Image>& normal);
    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W);

    std::shared_ptr<Tensor> getDst() const override { return dst; }

  protected:
    Image* getInput()
    {
      return color ? color.get() : (albedo ? albedo.get() : normal.get());
    }
  };

  class CPUInputReorderNode : public InputReorderNode
  {
  public:
    CPUInputReorderNode(const Ref<Device>& device,
                        const std::string& name,
                        const std::shared_ptr<Tensor>& dst,
                        const std::shared_ptr<TransferFunction>& transferFunc,
                        bool hdr,
                        bool snorm);

    void execute() override;
  };

} // namespace oidn
