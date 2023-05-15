// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "conv.h"

OIDN_NAMESPACE_BEGIN

  Conv::Conv(const ConvDesc& desc)
    : ConvDesc(desc)
  {
    if (srcDesc.getRank() != 3)
      throw std::invalid_argument("invalid convolution source shape");
    if (weightDesc.getRank() != 4 ||
        weightDesc.getI() != srcDesc.getC() ||
        weightDesc.getPaddedI() != srcDesc.getPaddedC())
      throw std::invalid_argument("invalid convolution weight shape");

    TensorDims dstDims;
    switch (postOp)
    {
    case PostOp::None:
      dstDims = {weightDesc.getO(), srcDesc.getH(), srcDesc.getW()};
      break;

    case PostOp::Pool:
      if (srcDesc.getH() % 2 != 0 || srcDesc.getW() % 2 != 0)
        throw std::invalid_argument("invalid pooling source shape");
      dstDims = {weightDesc.getO(), srcDesc.getH() / 2, srcDesc.getW() / 2};
      break;

    case PostOp::Upsample:
      dstDims = {weightDesc.getO(), srcDesc.getH() * 2, srcDesc.getW() * 2};
      break;

    default:
      throw std::invalid_argument("unsupported convolution postop");
    }

    TensorDims dstPaddedDims = dstDims;
    dstPaddedDims[0] = weightDesc.getPaddedO();

    dstDesc = {dstDims, dstPaddedDims, srcDesc.layout, srcDesc.dataType};

    if (!((biasDesc.getRank() == 1 && biasDesc.getX() == weightDesc.getO()
                                   && biasDesc.getPaddedX() == weightDesc.getPaddedO()) ||
          (biasDesc.getRank() == 3 && biasDesc.dims == dstDesc.dims
                                   && biasDesc.paddedDims == dstDesc.paddedDims)))
      throw std::invalid_argument("invalid convolution bias shape");
  }

  void Conv::setSrc(const std::shared_ptr<Tensor>& src)
  {
    if (!src || src->getDesc() != srcDesc)
      throw std::invalid_argument("invalid convolution source");

    this->src = src;
    updateSrc();
  }

  void Conv::setWeight(const std::shared_ptr<Tensor>& weight)
  {
    if (!weight || weight->getDesc() != weightDesc)
      throw std::invalid_argument("invalid convolution weight");

    this->weight = weight;
    updateWeight();
  }

  void Conv::setBias(const std::shared_ptr<Tensor>& bias)
  {
    if (!bias || bias->getDesc() != biasDesc)
      throw std::invalid_argument("invalid convolution bias");

    this->bias = bias;
    updateBias();
  }

  void Conv::setDst(const std::shared_ptr<Tensor>& dst)
  {
    if (!dst || dst->getDesc() != dstDesc)
      throw std::invalid_argument("invalid convolution destination");

    this->dst = dst;
    updateDst();
  }

OIDN_NAMESPACE_END
