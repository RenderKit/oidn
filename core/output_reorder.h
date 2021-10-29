// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"
#include "image.h"
#include "color.h"
#include "reorder.h"

namespace oidn {

  // Output reorder node
  class OutputReorderNode : public Node
  {
  protected:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Image> output;
    std::shared_ptr<TransferFunction> transferFunc;
    ReorderTile tile;
    bool hdr;
    bool snorm;

  public:
    OutputReorderNode(const Ref<Device>& device,
                      const std::string& name,
                      const std::shared_ptr<Tensor>& src,
                      const std::shared_ptr<TransferFunction>& transferFunc,
                      bool hdr,
                      bool snorm);

    void setDst(const std::shared_ptr<Image>& output);
    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W);
  };

  class CPUOutputReorderNode : public OutputReorderNode
  {
  public:
    CPUOutputReorderNode(const Ref<Device>& device,
                         const std::string& name,
                         const std::shared_ptr<Tensor>& src,
                         const std::shared_ptr<TransferFunction>& transferFunc,
                         bool hdr,
                         bool snorm);

    void execute() override;
  };

} // namespace oidn
