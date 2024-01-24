// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_reorder.h"

OIDN_NAMESPACE_BEGIN

  template<typename SrcT, typename DstT, TensorLayout srcLayout, TensorLayout dstLayout>
  bool tryReorderWeight(Tensor& src, int srcBeginI, int srcI, Tensor& dst, int dstBeginI, int dstI)
  {
    assert(srcBeginI + srcI <= src.getPaddedI());
    assert(dstBeginI + dstI <= dst.getPaddedI());

    if (src.getDataType() != DataTypeOf<SrcT>::value || src.getLayout() != srcLayout ||
        dst.getDataType() != DataTypeOf<DstT>::value || dst.getLayout() != dstLayout)
      return false;

    TensorAccessor4D<SrcT, srcLayout> srcAcc = src;
    TensorAccessor4D<DstT, dstLayout> dstAcc = dst;

    for (int o = 0; o < dstAcc.O; ++o)
    {
      for (int i = 0; i < dstI; ++i)
      {
        for (int h = 0; h < dstAcc.H; ++h)
        {
          for (int w = 0; w < dstAcc.W; ++w)
          {
            SrcT value;
            if (o < srcAcc.O && i < srcI)
              value = srcAcc(o, srcBeginI + i, h, w);
            else
              value = 0; // padding

            dstAcc(o, dstBeginI + i, h, w) = DstT(value);
          }
        }
      }
    }

    return true;
  }

  void reorderWeight(Tensor& src, int srcBeginI, int srcI, Tensor& dst, int dstBeginI, int dstI)
  {
    bool ok =
      tryReorderWeight<half, half,  TensorLayout::oihw, TensorLayout::oihw>        (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, float, TensorLayout::oihw, TensorLayout::oihw>        (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, half,  TensorLayout::oihw, TensorLayout::OIhw8i8o>    (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, float, TensorLayout::oihw, TensorLayout::OIhw8i8o>    (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, half,  TensorLayout::oihw, TensorLayout::OIhw16i16o>  (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, float, TensorLayout::oihw, TensorLayout::OIhw16i16o>  (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, half,  TensorLayout::oihw, TensorLayout::OIhw2o8i8o2i>(src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, half,  TensorLayout::oihw, TensorLayout::OIhw8i16o2i> (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, float, TensorLayout::oihw, TensorLayout::IOhw8i8o>    (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, float, TensorLayout::oihw, TensorLayout::IOhw16i16o>  (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, half,  TensorLayout::oihw, TensorLayout::ohwi>        (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, float, TensorLayout::oihw, TensorLayout::ohwi>        (src, srcBeginI, srcI, dst, dstBeginI, dstI);

    if (!ok)
      throw std::logic_error("unsupported weight layout or data type");
  }

  void reorderWeight(Tensor& src, Tensor& dst)
  {
    reorderWeight(src, 0, src.getI(), dst, 0, dst.getPaddedI());
  }

  template<typename SrcT, typename DstT>
  bool tryReorderBias(Tensor& src, Tensor& dst)
  {
    if (src.getDataType() != DataTypeOf<SrcT>::value ||
        dst.getDataType() != DataTypeOf<DstT>::value)
      return false;

    TensorAccessor1D<SrcT> srcAcc = src;
    TensorAccessor1D<DstT> dstAcc = dst;

    const int srcX = src.getX();

    for (int x = 0; x < srcX; ++x)
      dstAcc(x) = srcAcc(x);

    for (int x = srcX; x < dstAcc.X; ++x)
      dstAcc(x) = 0; // padding

    return true;
  }

  void reorderBias(Tensor& src, Tensor& dst)
  {
    bool ok = src.getLayout() == TensorLayout::x && dst.getLayout() == TensorLayout::x &&
      (tryReorderBias<half, half> (src, dst) ||
       tryReorderBias<half, float>(src, dst));

    if (!ok)
      throw std::logic_error("unsupported bias layout or data type");
  }

OIDN_NAMESPACE_END
