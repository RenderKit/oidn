// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor.h"

namespace oidn {

  namespace
  {
    template<typename SrcT, typename DstT, TensorLayout srcLayout, TensorLayout dstLayout>
    struct Reorder
    {
      void operator ()(const Tensor& src, Tensor& dst)
      {
        TensorAccessor4D<SrcT, srcLayout> srcAcc = src;
        TensorAccessor4D<DstT, dstLayout> dstAcc = dst;

        for (int o = 0; o < dstAcc.O; ++o)
        {
          for (int i = 0; i < dstAcc.I; ++i)
          {
            for (int h = 0; h < dstAcc.H; ++h)
            {
              for (int w = 0; w < dstAcc.W; ++w)
              {
                SrcT value;
                if (o < srcAcc.O && i < srcAcc.I)
                  value = srcAcc(o, i, h, w);
                else
                  value = 0; // padding

                dstAcc(o, i, h, w) = DstT(value);
              }
            }
          }
        }
      }
    };

    template<typename SrcT, typename DstT>
    struct Reorder<SrcT, DstT, TensorLayout::x, TensorLayout::x>
    {
      void operator ()(const Tensor& src, Tensor& dst)
      {
        TensorAccessor1D<SrcT> srcAcc = src;
        TensorAccessor1D<DstT> dstAcc = dst;

        for (int x = 0; x < srcAcc.X; ++x)
          dstAcc(x) = srcAcc(x);

        for (int x = srcAcc.X; x < dstAcc.X; ++x)
          dstAcc(x) = 0; // padding
      }
    };

    template<typename SrcT, typename DstT, TensorLayout srcLayout, TensorLayout dstLayout>
    bool tryReorder(const Tensor& src, Tensor& dst)
    {
      if (src.getDataType() == DataTypeOf<SrcT>::value && src.getLayout() == srcLayout &&
          dst.getDataType() == DataTypeOf<DstT>::value && dst.getLayout() == dstLayout)
      {
        Reorder<SrcT, DstT, srcLayout, dstLayout>()(src, dst);
        return true;
      }

      return false;
    }
  }

  void reorder(const Tensor& src, Tensor& dst)
  {
    bool ok =
      tryReorder<half, half,  TensorLayout::x,    TensorLayout::x>(src, dst) ||
      tryReorder<half, float, TensorLayout::x,    TensorLayout::x>(src, dst) ||
      tryReorder<half, half,  TensorLayout::oihw, TensorLayout::oihw>(src, dst) ||
      tryReorder<half, float, TensorLayout::oihw, TensorLayout::oihw>(src, dst) ||
      tryReorder<half, half,  TensorLayout::oihw, TensorLayout::OIhw8i8o>(src, dst) ||
      tryReorder<half, float, TensorLayout::oihw, TensorLayout::OIhw8i8o>(src, dst) ||
      tryReorder<half, half,  TensorLayout::oihw, TensorLayout::OIhw16i16o>(src, dst) ||
      tryReorder<half, float, TensorLayout::oihw, TensorLayout::OIhw16i16o>(src, dst) ||
      tryReorder<half, half,  TensorLayout::oihw, TensorLayout::OIhw2o8i8o2i>(src, dst) ||
      tryReorder<half, half,  TensorLayout::oihw, TensorLayout::OIhw8i16o2i>(src, dst) ||
      tryReorder<half, half,  TensorLayout::oihw, TensorLayout::ohwi>(src, dst) ||
      tryReorder<half, float, TensorLayout::oihw, TensorLayout::ohwi>(src, dst);

    if (!ok)
      throw std::logic_error("unsupported reorder");
  }

} // namespace oidn