// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/platform.h"

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
  struct TensorLayoutTraits<TensorLayout::x>
  {
    template<typename T>
    struct ByteOffset
    {
      static constexpr oidn_constant uint32_t xByteStride = sizeof(T);

      ByteOffset() = default;

      OIDN_HOST_DEVICE_INLINE uint32_t operator ()(int x) const
      {
        return uint32_t(x) * xByteStride;
      }
    };
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::chw>
  {
    template<typename T>
    struct ByteOffset
    {
      static constexpr oidn_constant uint32_t wByteStride = sizeof(T);
      uint32_t hByteStride;
      uint32_t cByteStride;

      ByteOffset() = default;

      OIDN_HOST_DEVICE_INLINE ByteOffset(int C, int H, int W)
      {
        hByteStride = uint32_t(W) * wByteStride;
        cByteStride = uint32_t(H) * hByteStride;
      }

      OIDN_HOST_DEVICE_INLINE uint32_t operator ()(int c, int h, int w) const
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
    struct ByteOffset
    {
      static constexpr oidn_constant uint32_t cByteStride = sizeof(T);
      uint32_t wByteStride;
      uint32_t hByteStride;

      ByteOffset() = default;

      OIDN_HOST_DEVICE_INLINE ByteOffset(int C, int H, int W)
      {
        wByteStride = uint32_t(C) * cByteStride;
        hByteStride = uint32_t(W) * wByteStride;
      }

      OIDN_HOST_DEVICE_INLINE uint32_t operator ()(int c, int h, int w) const
      {
        return uint32_t(c) * cByteStride +
               uint32_t(h) * hByteStride +
               uint32_t(w) * wByteStride;
      }
    };
  };

  template<typename T, int B>
  struct TensorByteOffsetChwBc
  {
    static constexpr oidn_constant int blockC = B; // block channels

    static constexpr oidn_constant uint32_t cByteStride = sizeof(T);
    static constexpr oidn_constant uint32_t wByteStride = B * cByteStride;
    uint32_t hByteStride;
    uint32_t CByteStride;

    TensorByteOffsetChwBc() = default;

    OIDN_HOST_DEVICE_INLINE TensorByteOffsetChwBc(int C, int H, int W)
    {
      hByteStride = uint32_t(W) * wByteStride;
      CByteStride = uint32_t(H) * hByteStride;
    }

    OIDN_HOST_DEVICE_INLINE uint32_t operator ()(int c, int h, int w) const
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
    using ByteOffset = TensorByteOffsetChwBc<T, 8>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::Chw16c>
  {
    template<typename T>
    using ByteOffset = TensorByteOffsetChwBc<T, 16>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::oihw>
  {
    template<typename T>
    struct ByteOffset
    {
      static constexpr oidn_constant uint32_t wByteStride = sizeof(T);
      uint32_t hByteStride;
      uint32_t iByteStride;
      uint32_t oByteStride;

      ByteOffset() = default;

      OIDN_HOST_DEVICE_INLINE ByteOffset(int O, int I, int H, int W)
      {
        hByteStride = uint32_t(W) * wByteStride;
        iByteStride = uint32_t(H) * hByteStride;
        oByteStride = uint32_t(I) * iByteStride;
      }

      OIDN_HOST_DEVICE_INLINE uint32_t operator ()(int o, int i, int h, int w) const
      {
        return uint32_t(o) * oByteStride +
               uint32_t(i) * iByteStride +
               uint32_t(h) * hByteStride +
               uint32_t(w) * wByteStride;
      }
    };
  };

  template<typename T, int B>
  struct TensorByteOffsetOIhwBiBo
  {
    static constexpr oidn_constant int blockC = B; // block channels

    static constexpr oidn_constant uint32_t BoByteStride = sizeof(T);
    static constexpr oidn_constant uint32_t BiByteStride = B * BoByteStride;
    static constexpr oidn_constant uint32_t wByteStride  = B * BiByteStride;
    uint32_t hByteStride;
    uint32_t IByteStride;
    uint32_t OByteStride;

    TensorByteOffsetOIhwBiBo() = default;

    OIDN_HOST_DEVICE_INLINE TensorByteOffsetOIhwBiBo(int O, int I, int H, int W)
    {
      hByteStride = uint32_t(W)     * wByteStride;
      IByteStride = uint32_t(H)     * hByteStride;
      OByteStride = uint32_t(I / B) * IByteStride;
    }

    OIDN_HOST_DEVICE_INLINE uint32_t operator ()(int o, int i, int h, int w) const
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
    using ByteOffset = TensorByteOffsetOIhwBiBo<T, 8>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::OIhw16i16o>
  {
    template<typename T>
    using ByteOffset = TensorByteOffsetOIhwBiBo<T, 16>;
  };

  template<typename T, int P, int Q, int R, int S>
  struct TensorByteOffsetOIhwPoQiRoSi
  {
    static_assert(P * R == Q * S, "invalid tensor layout parameters");

    static constexpr oidn_constant int B = P * R;
    static constexpr oidn_constant int blockC = B; // block channels

    static constexpr oidn_constant uint32_t SiByteStride = sizeof(T);
    static constexpr oidn_constant uint32_t RoByteStride = S * SiByteStride;
    static constexpr oidn_constant uint32_t QiByteStride = R * RoByteStride;
    static constexpr oidn_constant uint32_t PoByteStride = Q * QiByteStride;
    static constexpr oidn_constant uint32_t wByteStride  = P * PoByteStride;
    uint32_t hByteStride;
    uint32_t IByteStride;
    uint32_t OByteStride;

    TensorByteOffsetOIhwPoQiRoSi() = default;

    OIDN_HOST_DEVICE_INLINE TensorByteOffsetOIhwPoQiRoSi(int O, int I, int H, int W)
    {
      hByteStride = uint32_t(W)     * wByteStride;
      IByteStride = uint32_t(H)     * hByteStride;
      OByteStride = uint32_t(I / B) * IByteStride;
    }

    OIDN_HOST_DEVICE_INLINE uint32_t operator ()(int o, int i, int h, int w) const
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
    using ByteOffset = TensorByteOffsetOIhwPoQiRoSi<T, 2, 8, 8, 2>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::OIhw8i16o2i>
  {
    template<typename T>
    using ByteOffset = TensorByteOffsetOIhwPoQiRoSi<T, 1, 8, 16, 2>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::ohwi>
  {
    template<typename T>
    struct ByteOffset
    {
      static constexpr oidn_constant uint32_t iByteStride = sizeof(T);
      uint32_t wByteStride;
      uint32_t hByteStride;
      uint32_t oByteStride;

      ByteOffset() = default;

      OIDN_HOST_DEVICE_INLINE ByteOffset(int O, int I, int H, int W)
      {
        wByteStride = uint32_t(I) * iByteStride;
        hByteStride = uint32_t(W) * wByteStride;
        oByteStride = uint32_t(H) * hByteStride;
      }

      OIDN_HOST_DEVICE_INLINE uint32_t operator ()(int o, int i, int h, int w) const
      {
        return uint32_t(o) * oByteStride +
               uint32_t(i) * iByteStride +
               uint32_t(h) * hByteStride +
               uint32_t(w) * wByteStride;
      }
    };
  };

  template<typename T, TensorLayout layout>
  using TensorByteOffset = typename TensorLayoutTraits<layout>::template ByteOffset<T>;

  // -----------------------------------------------------------------------------------------------

#if !defined(OIDN_COMPILE_METAL_DEVICE)
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
#endif

OIDN_NAMESPACE_END
