// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"
#include "image.h"
#include "color.h"
#include "reorder.h"
#include "output_reorder_kernel.h"

namespace oidn {

  struct OutputReorderDesc
  {
    std::string name;
    std::shared_ptr<Tensor> src;
    std::shared_ptr<TransferFunction> transferFunc;
    bool hdr;
    bool snorm;
  };

  // Output reorder node
  class OutputReorderNode : public virtual Node
  {
  protected:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Image> output;
    std::shared_ptr<TransferFunction> transferFunc;
    ReorderTile tile;
    bool hdr;
    bool snorm;

  public:
    OutputReorderNode(const OutputReorderDesc& desc);

    void setDst(const std::shared_ptr<Image>& output);
    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W);
  };

  template<typename NodeT, typename TensorDT, TensorLayout tensorLayout>
  class XPUOutputReorderNode : public NodeT, public OutputReorderNode
  {
  public:
    XPUOutputReorderNode(const Ref<typename NodeT::DeviceType>& device, const OutputReorderDesc& desc)
      : NodeT(device, desc.name),
        OutputReorderNode(desc) {}

    void execute() override
    {
      switch (getDataType(output->format))
      {
      case DataType::Float32: executeKernel<float>(); break;
      case DataType::Float16: executeKernel<half>();  break;
      default:                assert(0);
      }
    }

  private:
    template<typename ImageDT>
    void executeKernel()
    {
      assert(tile.hSrcBegin + tile.H <= src->dims[1]);
      assert(tile.wSrcBegin + tile.W <= src->dims[2]);
      //assert(tile.hDstBegin + tile.H <= output->height);
      //assert(tile.wDstBegin + tile.W <= output->width);

      OutputReorder<ImageDT, TensorDT, tensorLayout> kernel;
      kernel.src = *src;
      kernel.output = *output;
      kernel.tile = tile;
      kernel.transferFunc = *transferFunc;
      kernel.hdr = hdr;
      kernel.snorm = snorm;

      this->device->executeKernel(tile.H, tile.W, kernel);
    }
  };

} // namespace oidn
