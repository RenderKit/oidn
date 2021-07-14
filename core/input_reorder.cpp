// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#if defined(OIDN_DEVICE_GPU)
  #include "sycl_device.h"
#endif

#include "input_reorder.h"
#include "input_reorder_ispc.h"

namespace oidn {

  InputReorderNode::InputReorderNode(const Ref<Device>& device,
                                     const std::string& name,
                                     const std::shared_ptr<Tensor>& dst,
                                     const std::shared_ptr<TransferFunction>& transferFunc,
                                     bool hdr,
                                     bool snorm)
    : Node(device, name),
      dst(dst),
      transferFunc(transferFunc),
      hdr(hdr),
      snorm(snorm)
  {
    assert(dst->ndims() == 3);
    assert(dst->layout == TensorLayout::chw ||
           dst->layout == TensorLayout::Chw8c ||
           dst->layout == TensorLayout::Chw16c);
    assert(dst->blockSize() == device->getTensorBlockSize());

    setTile(0, 0, 0, 0, 0, 0);
  }

  void InputReorderNode::setSrc(const std::shared_ptr<Image>& color, const std::shared_ptr<Image>& albedo, const std::shared_ptr<Image>& normal)
  {
    assert(dst->dims[0] >= (color  ? color->numChannels()  : 0) +
                           (albedo ? albedo->numChannels() : 0) +
                           (normal ? normal->numChannels() : 0));

    this->color  = color;
    this->albedo = albedo;
    this->normal = normal;
  }

  void InputReorderNode::setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W)
  {
    tile.hSrcBegin = hSrc;
    tile.wSrcBegin = wSrc;
    tile.hDstBegin = hDst;
    tile.wDstBegin = wDst;
    tile.H = H;
    tile.W = W;
  }

  CPUInputReorderNode::CPUInputReorderNode(const Ref<Device>& device,
                                           const std::string& name,
                                           const std::shared_ptr<Tensor>& dst,
                                           const std::shared_ptr<TransferFunction>& transferFunc,
                                           bool hdr,
                                           bool snorm)
    : InputReorderNode(device, name, dst, transferFunc, hdr, snorm) {}

  void CPUInputReorderNode::execute()
  {
    assert(tile.H + tile.hSrcBegin <= getHeight());
    assert(tile.W + tile.wSrcBegin <= getWidth());
    assert(tile.H + tile.hDstBegin <= dst->dims[1]);
    assert(tile.W + tile.wDstBegin <= dst->dims[2]);

    ispc::InputReorder impl;

    impl.color  = color  ? *color  : Image();
    impl.albedo = albedo ? *albedo : Image();
    impl.normal = normal ? *normal : Image();
    impl.dst = *dst;
    impl.tile = tile;
    impl.transferFunc = *transferFunc;
    impl.hdr = hdr;
    impl.snorm = snorm;

    parallel_nd(impl.dst.H, [&](int hDst)
    {
      ispc::InputReorder_kernel(&impl, hDst);
    });
  }

#if defined(OIDN_DEVICE_GPU)

  struct InputReorder
  {
    // Source
    ImageAccessor<float> color;
    ImageAccessor<float> albedo;
    ImageAccessor<float> normal;

    // Destination
    TensorAccessor<half> dst;

    // Tile
    ReorderTile tile;

    // Transfer function
    TransferFunction transferFunc;
    bool hdr;
    bool snorm; // signed normalized ([-1..1])

    __forceinline void storeZero(int h, int w, int c) const
    {
      dst.set(h, w, c, 0.f);
    }

    // Stores a color value
    __forceinline void storeColor(int h, int w, int c, vec3f value) const
    {
      // Scale
      value = value * transferFunc.getInputScale();

      // Sanitize
      value = clamp(nan_to_zero(value), snorm ? -1.f : 0.f, hdr ? std::numeric_limits<float>::max() : 1.f);

      if (snorm)
      {
        // Transform to [0..1]
        value = value * 0.5f + 0.5f;
      }

      // Apply the transfer function
      value = transferFunc.forward(value);

      // Store
      dst.set3(h, w, c, value);
    }

    // Stores an albedo value
    __forceinline void storeAlbedo(int h, int w, int c, vec3f value) const
    {
      // Scale
      if (!color.ptr)
        value = value * transferFunc.getInputScale();

      // Sanitize
      value = clamp(nan_to_zero(value), 0.f, 1.f);

      // Apply the transfer function
      if (!color.ptr)
        value = transferFunc.forward(value);

      // Store
      dst.set3(h, w, c, value);
    }

    // Stores a normal value
    __forceinline void storeNormal(int h, int w, int c, vec3f value) const
    {
      // Scale
      if (!color.ptr)
        value = value * transferFunc.getInputScale();

      // Sanitize
      value = clamp(nan_to_zero(value), -1.f, 1.f);

      // Transform to [0..1]
      value = value * 0.5f + 0.5f;

      // Store
      dst.set3(h, w, c, value);
    }

    __forceinline void operator()(int hDst, int wDst) const
    {
      const int h = hDst - tile.hDstBegin;
      const int w = wDst - tile.wDstBegin;

      if (h >= 0 && h < tile.H && w >= 0 && w < tile.W)
      {
        const int hSrc = h + tile.hSrcBegin;
        const int wSrc = w + tile.wSrcBegin;
        const int wDst = w + tile.wDstBegin;

        int c = 0;

        if (color.ptr)
        {
          storeColor(hDst, wDst, c, color.get3(hSrc, wSrc));
          c += 3;
        }

        if (albedo.ptr)
        {
          storeAlbedo(hDst, wDst, c, albedo.get3(hSrc, wSrc));
          c += 3;
        }

        if (normal.ptr)
        {
          storeNormal(hDst, wDst, c, normal.get3(hSrc, wSrc));
          c += 3;
        }

        for (; c < dst.C; ++c)
          storeZero(hDst, wDst, c);
      }
      else
      {
        // Zero pad
        for (int c = 0; c < dst.C; ++c)
          storeZero(hDst, wDst, c);
      }
    }
  };

  SYCLInputReorderNode::SYCLInputReorderNode(const Ref<SYCLDevice>& device,
                                             const std::string& name,
                                             const std::shared_ptr<Tensor>& dst,
                                             const std::shared_ptr<TransferFunction>& transferFunc,
                                             bool hdr,
                                             bool snorm)
    : InputReorderNode(device, name, dst, transferFunc, hdr, snorm) {}

  void SYCLInputReorderNode::execute()
  {
    assert(tile.H + tile.hSrcBegin <= getHeight());
    assert(tile.W + tile.wSrcBegin <= getWidth());
    assert(tile.H + tile.hDstBegin <= dst->dims[1]);
    assert(tile.W + tile.wDstBegin <= dst->dims[2]);

    InputReorder kernel;
    kernel.color  = color  ? *color  : Image();
    kernel.albedo = albedo ? *albedo : Image();
    kernel.normal = normal ? *normal : Image();
    kernel.dst = *dst;
    kernel.tile = tile;
    kernel.transferFunc = *transferFunc;
    kernel.hdr = hdr;
    kernel.snorm = snorm;

    auto queue = ((SYCLDevice*)getDevice())->getSYCLQueue();
    queue.parallel_for(sycl::range<2>(kernel.dst.H, kernel.dst.W), [=](sycl::id<2> idx) {
      kernel(int(idx[0]), int(idx[1]));
    });
  }

#endif

} // namespace oidn