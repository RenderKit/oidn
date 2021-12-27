// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"
#include "image.h"
#include "color.h"
#include "reorder.h"
#include "input_reorder_kernel.h"

namespace oidn {

  struct InputReorderDesc
  {
    std::string name;
    std::shared_ptr<Tensor> dst;
    std::shared_ptr<TransferFunction> transferFunc;
    bool hdr;
    bool snorm;
  };

  // Input reorder node
  class InputReorderNode : public virtual Node
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
    InputReorderNode(const InputReorderDesc& desc);

    void setSrc(const std::shared_ptr<Image>& color, const std::shared_ptr<Image>& albedo, const std::shared_ptr<Image>& normal);
    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W);

    std::shared_ptr<Tensor> getDst() const { return dst; }

  protected:
    Image* getInput()
    {
      return color ? color.get() : (albedo ? albedo.get() : normal.get());
    }
  };

  template<typename NodeT, typename TensorDT, TensorLayout tensorLayout>
  class XPUInputReorderNode : public NodeT, public InputReorderNode
  {
  public:
    XPUInputReorderNode(const Ref<typename NodeT::DeviceType>& device,
                        const InputReorderDesc& desc)
      : NodeT(device, desc.name),
        InputReorderNode(desc) {}

    void execute() override
    {
      switch (getDataType(getInput()->format))
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
      assert(tile.H + tile.hSrcBegin <= getInput()->height);
      assert(tile.W + tile.wSrcBegin <= getInput()->width);
      assert(tile.H + tile.hDstBegin <= dst->height());
      assert(tile.W + tile.wDstBegin <= dst->width());
      
      InputReorder<ImageDT, TensorDT, tensorLayout> kernel;
      kernel.color  = color  ? *color  : Image();
      kernel.albedo = albedo ? *albedo : Image();
      kernel.normal = normal ? *normal : Image();
      kernel.dst = *dst;
      kernel.tile = tile;
      kernel.transferFunc = *transferFunc;
      kernel.hdr = hdr;
      kernel.snorm = snorm;

      this->device->executeKernel(dst->height(), dst->width(), kernel);
    }
  };

} // namespace oidn
