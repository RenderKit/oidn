// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"

OIDN_NAMESPACE_BEGIN

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

  // -----------------------------------------------------------------------------------------------

  template<TensorLayout layout>
  struct TensorLayoutTraits;

  template<>
  struct TensorLayoutTraits<TensorLayout::chw>
  {
    template<typename T>
    struct Addressing
    {
      static constexpr uint32_t wByteStride = sizeof(T);
      uint32_t hByteStride;
      uint32_t cByteStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int C, int H, int W)
      {
        hByteStride = uint32_t(W) * wByteStride;
        cByteStride = uint32_t(H) * hByteStride;
      }

      OIDN_HOST_DEVICE_INLINE uint32_t getByteOffset(int c, int h, int w) const
      {
        return uint32_t(c) * cByteStride +
               uint32_t(h) * hByteStride +
               uint32_t(w) * wByteStride;
      }
    };
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::hwc>
  {
    template<typename T>
    struct Addressing
    {
      static constexpr uint32_t cByteStride = sizeof(T);
      uint32_t wByteStride;
      uint32_t hByteStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int C, int H, int W)
      {
        wByteStride = uint32_t(C) * cByteStride;
        hByteStride = uint32_t(W) * wByteStride;
      }

      OIDN_HOST_DEVICE_INLINE uint32_t getByteOffset(int c, int h, int w) const
      {
        return uint32_t(c) * cByteStride +
               uint32_t(h) * hByteStride +
               uint32_t(w) * wByteStride;
      }
    };
  };

  template<typename T, int B>
  struct TensorAddressingChwBc
  {
    static constexpr int blockC = B; // block channels

    static constexpr uint32_t cByteStride = sizeof(T);
    static constexpr uint32_t wByteStride = B * cByteStride;
    uint32_t hByteStride;
    uint32_t CByteStride;

    TensorAddressingChwBc() = default;

    OIDN_HOST_DEVICE_INLINE TensorAddressingChwBc(int C, int H, int W)
    {
      hByteStride = uint32_t(W) * wByteStride;
      CByteStride = uint32_t(H) * hByteStride;
    }

    OIDN_HOST_DEVICE_INLINE uint32_t getByteOffset(int c, int h, int w) const
    {
      return uint32_t(c/B) * CByteStride +
             uint32_t(h)   * hByteStride +
             uint32_t(w)   * wByteStride +
             uint32_t(c%B) * cByteStride;
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
      static constexpr uint32_t wByteStride = sizeof(T);
      uint32_t hByteStride;
      uint32_t iByteStride;
      uint32_t oByteStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int O, int I, int H, int W)
      {
        hByteStride = uint32_t(W) * wByteStride;
        iByteStride = uint32_t(H) * hByteStride;
        oByteStride = uint32_t(I) * iByteStride;
      }

      OIDN_HOST_DEVICE_INLINE uint32_t getByteOffset(int o, int i, int h, int w) const
      {
        return uint32_t(o) * oByteStride +
               uint32_t(i) * iByteStride +
               uint32_t(h) * hByteStride +
               uint32_t(w) * wByteStride;
      }
    };
  };

  template<typename T, int B>
  struct TensorAddressingOIhwBiBo
  {
    static constexpr int blockC = B; // block channels

    static constexpr uint32_t BoByteStride = sizeof(T);
    static constexpr uint32_t BiByteStride = B * BoByteStride;
    static constexpr uint32_t wByteStride  = B * BiByteStride;
    uint32_t hByteStride;
    uint32_t IByteStride;
    uint32_t OByteStride;

    TensorAddressingOIhwBiBo() = default;

    OIDN_HOST_DEVICE_INLINE TensorAddressingOIhwBiBo(int O, int I, int H, int W)
    {
      hByteStride = uint32_t(W)     * wByteStride;
      IByteStride = uint32_t(H)     * hByteStride;
      OByteStride = uint32_t(I / B) * IByteStride;
    }

    OIDN_HOST_DEVICE_INLINE uint32_t getByteOffset(int o, int i, int h, int w) const
    {
      return uint32_t(o / B) * OByteStride  +
             uint32_t(i / B) * IByteStride  +
             uint32_t(h)     * hByteStride  +
             uint32_t(w)     * wByteStride  +
             uint32_t(i % B) * BiByteStride +
             uint32_t(o % B) * BoByteStride;
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

    static constexpr uint32_t SiByteStride = sizeof(T);
    static constexpr uint32_t RoByteStride = S * SiByteStride;
    static constexpr uint32_t QiByteStride = R * RoByteStride;
    static constexpr uint32_t PoByteStride = Q * QiByteStride;
    static constexpr uint32_t wByteStride  = P * PoByteStride;
    uint32_t hByteStride;
    uint32_t IByteStride;
    uint32_t OByteStride;

    TensorAddressingOIhwPoQiRoSi() = default;

    OIDN_HOST_DEVICE_INLINE TensorAddressingOIhwPoQiRoSi(int O, int I, int H, int W)
    {
      hByteStride = uint32_t(W)     * wByteStride;
      IByteStride = uint32_t(H)     * hByteStride;
      OByteStride = uint32_t(I / B) * IByteStride;
    }

    OIDN_HOST_DEVICE_INLINE uint32_t getByteOffset(int o, int i, int h, int w) const
    {
      return uint32_t(o / B)     * OByteStride  +
             uint32_t(i / B)     * IByteStride  +
             uint32_t(h)         * hByteStride  +
             uint32_t(w)         * wByteStride  +
             uint32_t(o % B / R) * PoByteStride +
             uint32_t(i % B / S) * QiByteStride +
             uint32_t(o % R)     * RoByteStride +
             uint32_t(i % S)     * SiByteStride;
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
      static constexpr uint32_t iByteStride = sizeof(T);
      uint32_t wByteStride;
      uint32_t hByteStride;
      uint32_t oByteStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int O, int I, int H, int W)
      {
        wByteStride = uint32_t(I) * iByteStride;
        hByteStride = uint32_t(W) * wByteStride;
        oByteStride = uint32_t(H) * hByteStride;
      }

      OIDN_HOST_DEVICE_INLINE uint32_t getByteOffset(int o, int i, int h, int w) const
      {
        return uint32_t(o) * oByteStride +
               uint32_t(i) * iByteStride +
               uint32_t(h) * hByteStride +
               uint32_t(w) * wByteStride;
      }
    };
  };

  template<typename T, TensorLayout layout>
  using TensorAddressing = typename TensorLayoutTraits<layout>::template Addressing<T>;

  // -----------------------------------------------------------------------------------------------

  struct TensorLayoutInfo
  {
    int rank;
    int blockC;
  };

  // Returns information about the tensor layout
  OIDN_INLINE TensorLayoutInfo getTensorLayoutInfo(TensorLayout layout)
  {
    switch (layout)
    {
    case TensorLayout::x:
      return {1, 1};
    case TensorLayout::chw:
    case TensorLayout::hwc:
      return {3, 1};
    case TensorLayout::Chw8c:
      return {3, 8};
    case TensorLayout::Chw16c:
      return {3, 16};
    case TensorLayout::oihw:
    case TensorLayout::ohwi:
      return {4, 1};
    case TensorLayout::OIhw8i8o:
      return {4, 8};
    case TensorLayout::OIhw16i16o:
    case TensorLayout::OIhw2o8i8o2i:
    case TensorLayout::OIhw8i16o2i:
      return {4, 16};
    default:
      throw std::invalid_argument("invalid tensor layout");
    }
  }

OIDN_NAMESPACE_END
