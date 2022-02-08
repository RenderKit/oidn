// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "conv.h"
#include "pool.h"
#include "upsample.h"
#include "color.h"
#include "network.h"

namespace oidn {

  Network::Network(const Ref<Device>& device, const std::map<std::string, std::shared_ptr<Tensor>>& weightsMap)
    : device(device),
      blockSize(device->getTensorBlockSize()),
      weightsMap(weightsMap)
  {
  }

  void Network::run(Progress& progress)
  {
    for (size_t i = 0; i < ops.size(); ++i)
    {
      ops[i]->run();
      
      // Dump
      /*
      //device->wait();
      auto dst = ops[i]->getDst();
      if (dst)
        dst->dump("gpu/" + ops[i]->getName() + "_");
      */

      progress.update(1);
    }
  }

  double Network::getWorkAmount() const
  {
    return double(ops.size());
  }

  void Network::allocScratch(size_t size)
  {
    assert(!scratch);
    scratch = device->newScratchBuffer(size);
  }

  std::shared_ptr<Tensor> Network::newTensor(const TensorDesc& desc, ptrdiff_t offset)
  {
    assert(scratch);
    return scratch->newTensor(desc, offset);
  }

  std::shared_ptr<Image> Network::newImage(const ImageDesc& desc, ptrdiff_t offset)
  {
    assert(scratch);
    return scratch->newImage(desc, offset);
  }

  TensorDesc Network::getInputDesc(const TensorDims& srcDims, int alignment)
  {
    assert(srcDims.size() == 3); // CHW

    TensorDims dstDims = srcDims;
    dstDims[0] = round_up(srcDims[0], blockSize); // round up C
    dstDims[1] = round_up(srcDims[1], int64_t(alignment)); // round up H
    dstDims[2] = round_up(srcDims[2], int64_t(alignment)); // round up W

    return TensorDesc(dstDims, device->getTensorLayout(), device->getTensorDataType());
  }

  std::shared_ptr<InputProcess> Network::addInputProcess(const std::string& name,
                                                         const std::shared_ptr<Tensor>& dst,
                                                         const std::shared_ptr<TransferFunction>& transferFunc,
                                                         bool hdr,
                                                         bool snorm)
  {
    auto op = device->newInputProcess({dst, transferFunc, hdr, snorm});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

  std::shared_ptr<OutputProcess> Network::addOutputProcess(const std::string& name,
                                                           const std::shared_ptr<Tensor>& src,
                                                           const std::shared_ptr<TransferFunction>& transferFunc,
                                                           bool hdr,
                                                           bool snorm)
  {
    auto op = device->newOutputProcess({src, transferFunc, hdr, snorm});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

  TensorDesc Network::getConvDesc(const std::string& name, const TensorDesc& srcDesc)
  {
    assert(srcDesc.getRank() == 3); // CHW

    const auto& bias = weightsMap[name + ".bias"];
    TensorDims dstDims = srcDesc.dims;
    dstDims[0] = round_up(bias->getX(), blockSize); // dstDims[C] = round_up(OC, blockSize)
    return TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
  }

  std::shared_ptr<Conv> Network::addConv(const std::string& name,
                                           const std::shared_ptr<Tensor>& src,
                                           const std::shared_ptr<Tensor>& dst,
                                           bool relu)
  {
    assert(dst->getDesc() == getConvDesc(name, src->getDesc()));

    // Get and reorder/pad the weight and bias tensors
    auto weight = reorderWeight(weightsMap[name + ".weight"]);
    auto bias = reorderBias(weightsMap[name + ".bias"]);

    // Create the convolution op
    auto op = device->newConv({src, weight, bias, dst, relu});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

   std::shared_ptr<Op> Network::addConcatConv(const std::string& name,
                                              const std::shared_ptr<Tensor>& src1,
                                              const std::shared_ptr<Tensor>& src2,
                                              const std::shared_ptr<Tensor>& dst,
                                              bool relu)
  {
    // Get and reorder/pad the weight and bias tensors
    auto weight = reorderWeight(weightsMap[name + ".weight"]);
    auto bias = reorderBias(weightsMap[name + ".bias"]);

    std::shared_ptr<Op> op;
    if (device->getTensorLayout() == TensorLayout::hwc)
    {
      // Concatenation is non-trivial -> create fused concatenation + convolution op
      op = device->newConcatConv({src1, src2, weight, bias, dst, relu});
    }
    else
    {
      // Concatenation is trivial (no-op) -> create convolution op
      if (src1->getBuffer() != src2->getBuffer() || (src1->getBufferOffset() + src1->getByteSize()) != src2->getBufferOffset())
        throw std::logic_error("concatenation is non-trivial");
      TensorDesc srcDesc = getConcatDesc({src1->getDesc(), src2->getDesc()});
      auto src = newTensor(srcDesc, src1->getBufferOffset());
      op = device->newConv({src, weight, bias, dst, relu});
    }

    op->setName(name);
    ops.push_back(op);
    return op;
  }

  TensorDesc Network::getPoolDesc(const TensorDesc& srcDesc)
  {
    assert(srcDesc.getRank() == 3); // CHW

    TensorDims dstDims = srcDesc.dims;
    dstDims[1] /= 2; // H/2
    dstDims[2] /= 2; // W/2
    return TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
  }

  std::shared_ptr<Pool> Network::addPool(const std::string& name,
                                         const std::shared_ptr<Tensor>& src,
                                         const std::shared_ptr<Tensor>& dst)
  {
    assert(dst->getDesc() == getPoolDesc(src->getDesc()));

    auto op = device->newPool({src, dst});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

  TensorDesc Network::getUpsampleDesc(const TensorDesc& srcDesc)
  {
    assert(srcDesc.getRank() == 3); // CHW

    TensorDims dstDims = srcDesc.dims;
    dstDims[1] *= 2; // H*2
    dstDims[2] *= 2; // W*2
    return TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
  }

  std::shared_ptr<Upsample> Network::addUpsample(const std::string& name,
                                                 const std::shared_ptr<Tensor>& src,
                                                 const std::shared_ptr<Tensor>& dst)
  {
    assert(dst->getDesc() == getUpsampleDesc(src->getDesc()));

    auto op = device->newUpsample({src, dst});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

  TensorDesc Network::getConcatDesc(const std::vector<TensorDesc>& srcDescs)
  {
    assert(!srcDescs.empty());
    assert(srcDescs[0].getRank() == 3); // CHW

    TensorDims dstDims = srcDescs[0].dims;
    for (size_t i = 1; i < srcDescs.size(); ++i)
    {
      assert(srcDescs[i].getRank() == 3); // CHW
      assert(srcDescs[i].dims[1] == srcDescs[0].dims[1]); // H
      assert(srcDescs[i].dims[2] == srcDescs[0].dims[2]); // W
      assert(srcDescs[i].layout == srcDescs[0].layout);
      assert(srcDescs[i].dataType == srcDescs[0].dataType);
      dstDims[0] += srcDescs[i].dims[0]; // C
    }
    return TensorDesc(dstDims, srcDescs[0].layout, srcDescs[0].dataType);
  }

  void Network::finalize()
  {
    // Compute the size of the scratch memory for the ops
    size_t opScratchSize = 0;
    for (const auto& op : ops)
      opScratchSize = max(opScratchSize, op->getScratchSize());

    // Allocate the scratch memory for the ops
    TensorDims opScratchDims = { int64_t(opScratchSize) };
    auto opScratch = device->newTensor({opScratchDims, TensorLayout::x, DataType::UInt8});

    // Set the scratch memory for the ops
    for (auto& op : ops)
      op->setScratch(opScratch);

    // Free the weights
    weightsMap.clear();

    // Print statistics
    if (device->isVerbose(2))
    {
      const size_t scratchSize = scratch ? scratch->getByteSize() : 0;
      const size_t totalScratchSize = scratchSize + opScratchSize;
      std::cout << "Tensor scratch bytes   : " << scratchSize << std::endl;
      std::cout << "Operation scratch bytes: " << opScratchSize << std::endl;
      std::cout << "Total scratch bytes    : " << totalScratchSize << std::endl;
    }
  }

  std::shared_ptr<Tensor> Network::reorderWeight(const std::shared_ptr<Tensor>& src)
  {
    if (src->getRank() != 4)
      throw Exception(Error::InvalidOperation, "invalid convolution weight");  

    const int O = round_up(src->getO(), blockSize);
    const int I = round_up(src->getI(), blockSize);
    const int H = src->getH();
    const int W = src->getW();

    auto dst = device->newTensor({{O, I, H, W}, device->getWeightsLayout(), device->getTensorDataType()});
    reorder(*src, *dst);
    return dst;
  }

  std::shared_ptr<Tensor> Network::reorderBias(const std::shared_ptr<Tensor>& src)
  {
    if (src->getRank() != 1)
      throw Exception(Error::InvalidOperation, "invalid convolution biases");

    const int X = round_up(src->getX(), blockSize);

    auto dst = device->newTensor({{X}, TensorLayout::x, device->getTensorDataType()});
    reorder(*src, *dst);
    return dst;
  }

} // namespace oidn
