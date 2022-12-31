// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

namespace oidn {

  // Tensor memory layout
  enum class TensorLayout
  {
    x,
    
    chw,
    Chw8c,  // blocked
    Chw16c, // blocked
    oihw,
    OIhw8i8o,     // blocked
    OIhw16i16o,   // blocked
    OIhw2o8i8o2i, // blocked (Xe-HPG DPAS)
    OIhw8i16o2i,  // blocked (Xe-HPC DPAS)

    hwc,
    ohwi,
  };

  template<TensorLayout layout>
  struct TensorLayoutTraits;

  template<>
  struct TensorLayoutTraits<TensorLayout::chw>
  {
    template<typename T>
    struct Addressing
    {
      static constexpr size_t wByteStride = sizeof(T);
      size_t hByteStride;
      size_t cByteStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int C, int H, int W)
      {
        hByteStride = size_t(W) * wByteStride;
        cByteStride = size_t(H) * hByteStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int c, int h, int w) const
      {
        return size_t(c) * cByteStride +
               size_t(h) * hByteStride +
               size_t(w) * wByteStride;
      }
    };
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::hwc>
  {
    template<typename T>
    struct Addressing
    {
      static constexpr size_t cByteStride = sizeof(T);
      size_t wByteStride;
      size_t hByteStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int C, int H, int W)
      {
        wByteStride = size_t(C) * cByteStride;
        hByteStride = size_t(W) * wByteStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int c, int h, int w) const
      {
        return size_t(c) * cByteStride +
               size_t(h) * hByteStride +
               size_t(w) * wByteStride;
      }
    };
  };

  template<typename T, int B>
  struct TensorAddressingChwBc
  {
    static constexpr int blockC = B; // block channels

    static constexpr size_t wByteStride = B * sizeof(T);
    size_t hByteStride;
    size_t CByteStride;

    TensorAddressingChwBc() = default;

    OIDN_HOST_DEVICE_INLINE TensorAddressingChwBc(int C, int H, int W)
    {
      hByteStride = size_t(W) * wByteStride;
      CByteStride = size_t(H) * hByteStride;
    }

    OIDN_HOST_DEVICE_INLINE size_t getOffset(int c, int h, int w) const
    {
      return size_t(c/B) * CByteStride +
             size_t(h)   * hByteStride +
             size_t(w)   * wByteStride +
             size_t(c%B) * sizeof(T);
    }
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::Chw8c>
  {
    template<typename T>
    using Addressing = TensorAddressingChwBc<T, 8>;
  };
  
  template<>
  struct TensorLayoutTraits<TensorLayout::Chw16c>
  {
    template<typename T>
    using Addressing = TensorAddressingChwBc<T, 16>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::oihw>
  {
    template<typename T>
    struct Addressing
    {
      static constexpr size_t wByteStride = sizeof(T);
      size_t hByteStride;
      size_t iByteStride;
      size_t oByteStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int O, int I, int H, int W)
      {
        hByteStride = size_t(W) * wByteStride;
        iByteStride = size_t(H) * hByteStride;
        oByteStride = size_t(I) * iByteStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int o, int i, int h, int w) const
      {
        return size_t(o) * oByteStride +
               size_t(i) * iByteStride +
               size_t(h) * hByteStride +
               size_t(w) * wByteStride;
      }
    };
  };

  template<typename T, int B>
  struct TensorAddressingOIhwBiBo
  {
    static constexpr int blockC = B; // block channels

    static constexpr size_t BoByteStride = sizeof(T);
    static constexpr size_t BiByteStride = B * BoByteStride;
    static constexpr size_t wByteStride  = B * BiByteStride;
    size_t hByteStride;
    size_t IByteStride;
    size_t OByteStride;

    TensorAddressingOIhwBiBo() = default;

    OIDN_HOST_DEVICE_INLINE TensorAddressingOIhwBiBo(int O, int I, int H, int W)
    {
      hByteStride = size_t(W)     * wByteStride;
      IByteStride = size_t(H)     * hByteStride;
      OByteStride = size_t(I / B) * IByteStride;
    }

    OIDN_HOST_DEVICE_INLINE size_t getOffset(int o, int i, int h, int w) const
    {
      return size_t(o / B) * OByteStride  +
             size_t(i / B) * IByteStride  +
             size_t(h)     * hByteStride  +
             size_t(w)     * wByteStride  +
             size_t(i % B) * BiByteStride +
             size_t(o % B) * BoByteStride;
    }
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::OIhw8i8o>
  {
    template<typename T>
    using Addressing = TensorAddressingOIhwBiBo<T, 8>;
  };
  
  template<>
  struct TensorLayoutTraits<TensorLayout::OIhw16i16o>
  {
    template<typename T>
    using Addressing = TensorAddressingOIhwBiBo<T, 16>;
  };

  template<typename T, int P, int Q, int R, int S>
  struct TensorAddressingOIhwPoQiRoSi
  {
    static_assert(P * R == Q * S, "invalid tensor layout parameters");

    static constexpr int B = P * R;
    static constexpr int blockC = B; // block channels

    static constexpr size_t SiByteStride = sizeof(T);
    static constexpr size_t RoByteStride = S * SiByteStride;
    static constexpr size_t QiByteStride = R * RoByteStride;
    static constexpr size_t PoByteStride = Q * QiByteStride;
    static constexpr size_t wByteStride  = P * PoByteStride;
    size_t hByteStride;
    size_t IByteStride;
    size_t OByteStride;

    TensorAddressingOIhwPoQiRoSi() = default;

    OIDN_HOST_DEVICE_INLINE TensorAddressingOIhwPoQiRoSi(int O, int I, int H, int W)
    {
      hByteStride = size_t(W)     * wByteStride;
      IByteStride = size_t(H)     * hByteStride;
      OByteStride = size_t(I / B) * IByteStride;
    }

    OIDN_HOST_DEVICE_INLINE size_t getOffset(int o, int i, int h, int w) const
    {
      return size_t(o / B)     * OByteStride  +
             size_t(i / B)     * IByteStride  +
             size_t(h)         * hByteStride  +
             size_t(w)         * wByteStride  +
             size_t(o % B / R) * PoByteStride +
             size_t(i % B / S) * QiByteStride +
             size_t(o % R)     * RoByteStride +
             size_t(i % S)     * SiByteStride;
    }
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::OIhw2o8i8o2i>
  {
    template<typename T>
    using Addressing = TensorAddressingOIhwPoQiRoSi<T, 2, 8, 8, 2>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::OIhw8i16o2i>
  {
    template<typename T>
    using Addressing = TensorAddressingOIhwPoQiRoSi<T, 1, 8, 16, 2>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::ohwi>
  {
    template<typename T>
    struct Addressing
    {
      static constexpr size_t iByteStride = sizeof(T);
      size_t wByteStride;
      size_t hByteStride;
      size_t oByteStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int O, int I, int H, int W)
      {
        wByteStride = size_t(I) * iByteStride;
        hByteStride = size_t(W) * wByteStride;
        oByteStride = size_t(H) * hByteStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int o, int i, int h, int w) const
      {
        return size_t(o) * oByteStride +
               size_t(i) * iByteStride +
               size_t(h) * hByteStride +
               size_t(w) * wByteStride;
      }
    };
  };

  template<typename T, TensorLayout layout>
  using TensorAddressing = typename TensorLayoutTraits<layout>::template Addressing<T>;

} // namespace oidn
