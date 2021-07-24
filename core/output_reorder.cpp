// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#if defined(OIDN_DEVICE_GPU)
  #include "sycl_device.h"
#endif

#include "output_reorder.h"
#include "output_reorder_ispc.h"

namespace oidn {
  
  OutputReorderNode::OutputReorderNode(const Ref<Device>& device,
                                       const std::string& name,
                                       const std::shared_ptr<Tensor>& src,
                                       const std::shared_ptr<TransferFunction>& transferFunc,
                                       bool hdr,
                                       bool snorm)
    : Node(device, name),
      src(src),
      transferFunc(transferFunc),
      hdr(hdr),
      snorm(snorm)
  {
    assert(src->ndims() == 3);
    assert(src->layout == TensorLayout::chw ||
           src->layout == TensorLayout::Chw8c ||
           src->layout == TensorLayout::Chw16c);
    assert(src->blockSize() == device->getTensorBlockSize());

    setTile(0, 0, 0, 0, 0, 0);
  }

  void OutputReorderNode::setDst(const std::shared_ptr<Image>& output)
  {
    assert(output);
    assert(src->dims[0] >= output->numChannels());
    assert(output->numChannels() == 3);

    this->output = output;
  }

  void OutputReorderNode::setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W)
  {
    tile.hSrcBegin = hSrc;
    tile.wSrcBegin = wSrc;
    tile.hDstBegin = hDst;
    tile.wDstBegin = wDst;
    tile.H = H;
    tile.W = W;
  }

  CPUOutputReorderNode::CPUOutputReorderNode(const Ref<Device>& device,
                                             const std::string& name,
                                             const std::shared_ptr<Tensor>& src,
                                             const std::shared_ptr<TransferFunction>& transferFunc,
                                             bool hdr,
                                             bool snorm)
    : OutputReorderNode(device, name, src, transferFunc, hdr, snorm) {}

  void CPUOutputReorderNode::execute()
  {
    assert(tile.hSrcBegin + tile.H <= src->dims[1]);
    assert(tile.wSrcBegin + tile.W <= src->dims[2]);
    //assert(tile.hDstBegin + tile.H <= output->height);
    //assert(tile.wDstBegin + tile.W <= output->width);

    ispc::OutputReorder impl;

    impl.src = *src;
    impl.output = *output;
    impl.tile = tile;
    impl.transferFunc = *transferFunc;
    impl.hdr = hdr;
    impl.snorm = snorm;

    parallel_nd(impl.tile.H, [&](int h)
    {
      ispc::OutputReorder_kernel(&impl, h);
    });
  }

#if defined(OIDN_DEVICE_GPU)

  template<typename T>
  struct OutputReorder
  {
    // Source
    TensorAccessor<half> src;

    // Destination
    ImageAccessor<T> output;

    // Tile
    ReorderTile tile;

    // Transfer function
    TransferFunction transferFunc;
    bool hdr;
    bool snorm; // signed normalized ([-1..1])

    __forceinline void operator()(int h, int w) const
    {
      const int hSrc = h + tile.hSrcBegin;
      const int hDst = h + tile.hDstBegin;
      const int wSrc = w + tile.wSrcBegin;
      const int wDst = w + tile.wDstBegin;

      // Load
      vec3f value = src.get3(hSrc, wSrc, 0);

      // The CNN output may contain negative values or even NaNs, so it must be sanitized
      value = clamp(nan_to_zero(value), 0.f, std::numeric_limits<float>::max());

      // Apply the inverse transfer function
      value = transferFunc.inverse(value);

      // Sanitize
      if (snorm)
      {
        // Transform to [-1..1]
        value = value * 2.f - 1.f;
        value = max(value, -1.f);
      }
      if (!hdr)
        value = min(value, 1.f);

      // Scale
      value = value * transferFunc.getOutputScale();

      // Store
      output.set3(hDst, wDst, value);
    }
  };

  SYCLOutputReorderNode::SYCLOutputReorderNode(const Ref<SYCLDevice>& device,
                                               const std::string& name,   
                                               const std::shared_ptr<Tensor>& src,
                                               const std::shared_ptr<TransferFunction>& transferFunc,
                                               bool hdr,
                                               bool snorm)
    : OutputReorderNode(device, name, src, transferFunc, hdr, snorm) {}

  void SYCLOutputReorderNode::execute()
  {
    switch (getDataType(output->format))
    {
    case DataType::Float32: executeKernel<float>(); break;
    case DataType::Float16: executeKernel<half>();  break;
    default:                assert(0);
    }
  }

  template<typename T>
  void SYCLOutputReorderNode::executeKernel()
  {
    assert(tile.hSrcBegin + tile.H <= src->dims[1]);
    assert(tile.wSrcBegin + tile.W <= src->dims[2]);
    //assert(tile.hDstBegin + tile.H <= output->height);
    //assert(tile.wDstBegin + tile.W <= output->width);

    OutputReorder<T> kernel;
    kernel.src = *src;
    kernel.output = *output;
    kernel.tile = tile;
    kernel.transferFunc = *transferFunc;
    kernel.hdr = hdr;
    kernel.snorm = snorm;

    auto& queue = ((SYCLDevice*)getDevice())->getSYCLQueue();
    queue.parallel_for(sycl::range<2>(tile.H, tile.W), [=](sycl::id<2> idx) {
      kernel(int(idx[0]), int(idx[1]));
    });
  }

#endif

} // namespace oidn