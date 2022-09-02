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
    OIhw8i8o,   // blocked
    OIhw16i16o, // blocked
    OIhw2o8i8o2i, // blocked (DG2)
    OIhw8i16o2i, // blocked (PVC)

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
      static constexpr size_t wStride = sizeof(T);
      size_t hStride;
      size_t cStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int C, int H, int W)
      {
        hStride = size_t(W) * wStride;
        cStride = size_t(H) * hStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int c, int h, int w) const
      {
        return size_t(c) * cStride +
               size_t(h) * hStride +
               size_t(w) * wStride;
      }
    };
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::hwc>
  {
    template<typename T>
    struct Addressing
    {
      static constexpr size_t cStride = sizeof(T);
      size_t wStride;
      size_t hStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int C, int H, int W)
      {
        wStride = size_t(C) * cStride;
        hStride = size_t(W) * wStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int c, int h, int w) const
      {
        return size_t(c) * cStride +
               size_t(h) * hStride +
               size_t(w) * wStride;
      }
    };
  };

  template<typename T, int B>
  struct TensorAddressingChwBc
  {
    static constexpr int blockC = B; // block channels

    static constexpr size_t wStride = B * sizeof(T);
    size_t hStride;
    size_t CStride;

    TensorAddressingChwBc() = default;

    OIDN_HOST_DEVICE_INLINE TensorAddressingChwBc(int C, int H, int W)
    {
      hStride = size_t(W) * wStride;
      CStride = size_t(H) * hStride;
    }

    OIDN_HOST_DEVICE_INLINE size_t getOffset(int c, int h, int w) const
    {
      return size_t(c/B) * CStride +
             size_t(h)   * hStride +
             size_t(w)   * wStride +
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
      static constexpr size_t wStride = sizeof(T);
      size_t hStride;
      size_t iStride;
      size_t oStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int O, int I, int H, int W)
      {
        hStride = size_t(W) * wStride;
        iStride = size_t(H) * hStride;
        oStride = size_t(I) * iStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int o, int i, int h, int w) const
      {
        return size_t(o) * oStride +
               size_t(i) * iStride +
               size_t(h) * hStride +
               size_t(w) * wStride;
      }
    };
  };

  template<typename T, int B>
  struct TensorAddressingOIhwBiBo
  {
    static constexpr int blockC = B; // block channels

    static constexpr size_t BoStride = sizeof(T);
    static constexpr size_t BiStride = B * BoStride;
    static constexpr size_t wStride  = B * BiStride;
    size_t hStride;
    size_t IStride;
    size_t OStride;

    TensorAddressingOIhwBiBo() = default;

    OIDN_HOST_DEVICE_INLINE TensorAddressingOIhwBiBo(int O, int I, int H, int W)
    {
      hStride = size_t(W)     * wStride;
      IStride = size_t(H)     * hStride;
      OStride = size_t(I / B) * IStride;
    }

    OIDN_HOST_DEVICE_INLINE size_t getOffset(int o, int i, int h, int w) const
    {
      return size_t(o / B) * OStride  +
             size_t(i / B) * IStride  +
             size_t(h)     * hStride  +
             size_t(w)     * wStride  +
             size_t(i % B) * BiStride +
             size_t(o % B) * BoStride;
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

    static constexpr size_t SiStride = sizeof(T);
    static constexpr size_t RoStride = S * SiStride;
    static constexpr size_t QiStride = R * RoStride;
    static constexpr size_t PoStride = Q * QiStride;
    static constexpr size_t wStride  = P * PoStride;
    size_t hStride;
    size_t IStride;
    size_t OStride;

    TensorAddressingOIhwPoQiRoSi() = default;

    OIDN_HOST_DEVICE_INLINE TensorAddressingOIhwPoQiRoSi(int O, int I, int H, int W)
    {
      hStride = size_t(W)     * wStride;
      IStride = size_t(H)     * hStride;
      OStride = size_t(I / B) * IStride;
    }

    OIDN_HOST_DEVICE_INLINE size_t getOffset(int o, int i, int h, int w) const
    {
      return size_t(o / B)     * OStride  +
             size_t(i / B)     * IStride  +
             size_t(h)         * hStride  +
             size_t(w)         * wStride  +
             size_t(o % B / R) * PoStride +
             size_t(i % B / S) * QiStride +
             size_t(o % R)     * RoStride +
             size_t(i % S)     * SiStride;
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
      static constexpr size_t iStride = sizeof(T);
      size_t wStride;
      size_t hStride;
      size_t oStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int O, int I, int H, int W)
      {
        wStride = size_t(I) * iStride;
        hStride = size_t(W) * wStride;
        oStride = size_t(H) * hStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int o, int i, int h, int w) const
      {
        return size_t(o) * oStride +
               size_t(i) * iStride +
               size_t(h) * hStride +
               size_t(w) * wStride;
      }
    };
  };

  template<typename T, TensorLayout layout>
  using TensorAddressing = typename TensorLayoutTraits<layout>::template Addressing<T>;

} // namespace oidn
