// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "concat_conv.h"

OIDN_NAMESPACE_BEGIN

  ConcatConv::ConcatConv(const ConcatConvDesc& desc)
    : ConcatConvDesc(desc)
  {
    if (src1Desc.getRank() != 3 ||
        src2Desc.getRank() != 3 ||
        src1Desc.getH() != src2Desc.getH() ||
        src1Desc.getW() != src2Desc.getW() ||
        src1Desc.layout != src2Desc.layout ||
        src1Desc.dataType != src2Desc.dataType)
      throw std::invalid_argument("invalid concat+conv source descriptor");
    if (weightDesc.getRank() != 4 || weightDesc.getI() != (src1Desc.getC() + src2Desc.getC()) ||
        weightDesc.getPaddedI() != (src1Desc.getPaddedC() + src2Desc.getPaddedC()))
      throw std::invalid_argument("invalid concat+conv weight shape");

    TensorDims dstDims{weightDesc.getO(), src1Desc.getH(), src1Desc.getW()};
    TensorDims dstPaddedDims{weightDesc.getPaddedO(), src1Desc.getH(), src1Desc.getW()};
    dstDesc = {dstDims, dstPaddedDims, src1Desc.layout, src1Desc.dataType};
  }

  void ConcatConv::setSrc(const Ref<Tensor>& src1, const Ref<Tensor>& src2)
  {
    if (!src1 || src1->getDesc() != src1Desc || !src2 || src2->getDesc() != src2Desc)
      throw std::invalid_argument("invalid concat+conv source");

    this->src1 = src1;
    this->src2 = src2;
    updateSrc();
  }

  void ConcatConv::setBias(const Ref<Tensor>& bias)
  {
    if (!bias || bias->getDesc() != biasDesc)
      throw std::invalid_argument("invalid concat+conv bias");

    this->bias = bias;
    updateBias();
  }

  void ConcatConv::setDst(const Ref<Tensor>& dst)
  {
    if (!dst || dst->getDesc() != dstDesc)
      throw std::invalid_argument("invalid concat+conv destination");

    this->dst = dst;
    updateDst();
  }

OIDN_NAMESPACE_END
