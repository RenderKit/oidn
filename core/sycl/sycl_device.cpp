// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mkl-dnn/include/dnnl_sycl.hpp"
#include "../input_reorder.h"
#include "../output_reorder.h"
#include "../image_copy.h"
#include "sycl_device.h"
#include "sycl_buffer.h"
#include "sycl_node.h"
#include "sycl_pool.h"
#include "sycl_upsample.h"

namespace oidn {

  SYCLDevice::SYCLDevice() {}

  SYCLDevice::SYCLDevice(const sycl::queue& syclQueue)
    : sycl(new SYCL{syclQueue.get_context(), syclQueue.get_device(), syclQueue}) {}

  void SYCLDevice::init()
  {
    // Initialize the neural network runtime
    dnnl_set_verbose(clamp(verbose - 2, 0, 2)); // unfortunately this is not per-device but global

    if (sycl)
    {
      if (!sycl->device.is_gpu())
        throw Exception(Error::InvalidArgument, "unsupported SYCL device");
      if (!sycl->queue.is_in_order())
        throw Exception(Error::InvalidArgument, "unsupported out-of-order SYCL queue");

      dnnlEngine = dnnl::sycl_interop::make_engine(sycl->device, sycl->context);
      dnnlStream = dnnl::sycl_interop::make_stream(dnnlEngine, sycl->queue);
    }
    else
    {
      dnnlEngine  = dnnl::engine(dnnl::engine::kind::gpu, 0);
      dnnlStream  = dnnl::stream(dnnlEngine, dnnl::stream::flags::in_order);
      sycl.reset(new SYCL{dnnl::sycl_interop::get_context(dnnlEngine),
                          dnnl::sycl_interop::get_device(dnnlEngine),
                          dnnl::sycl_interop::get_queue(dnnlStream)});
    }

    tensorDataType = DataType::Float16;
    tensorBlockSize = 16;
  }

  void SYCLDevice::printInfo()
  {
    std::cout << "  Device  : " << sycl->device.get_info<sycl::info::device::name>() << std::endl;

    std::cout << "  Neural  : ";
    std::cout << "DNNL (oneDNN) " << DNNL_VERSION_MAJOR << "." <<
                                     DNNL_VERSION_MINOR << "." <<
                                     DNNL_VERSION_PATCH;
    std::cout << std::endl;
  }

  Ref<Buffer> SYCLDevice::newBuffer(size_t byteSize, Buffer::Kind kind)
  {
    return makeRef<SYCLBuffer>(Ref<SYCLDevice>(this), byteSize, kind);
  }

  Ref<Buffer> SYCLDevice::newBuffer(void* ptr, size_t byteSize)
  {
    return makeRef<SYCLBuffer>(Ref<SYCLDevice>(this), ptr, byteSize);
  }

  std::shared_ptr<PoolNode> SYCLDevice::newPoolNode(const PoolDesc& desc)
  {
    return std::make_shared<SYCLPoolNode>(Ref<SYCLDevice>(this), desc);
  }

  std::shared_ptr<UpsampleNode> SYCLDevice::newUpsampleNode(const UpsampleDesc& desc)
  {
    return std::make_shared<SYCLUpsampleNode>(Ref<SYCLDevice>(this), desc);
  }

  std::shared_ptr<InputReorderNode> SYCLDevice::newInputReorderNode(const InputReorderDesc& desc)
  {
    return std::make_shared<XPUInputReorderNode<SYCLNode, half, TensorLayout::Chw16c>>(Ref<SYCLDevice>(this), desc);
  }

  std::shared_ptr<OutputReorderNode> SYCLDevice::newOutputReorderNode(const OutputReorderDesc& desc)
  {
    return std::make_shared<XPUOutputReorderNode<SYCLNode, half, TensorLayout::Chw16c>>(Ref<SYCLDevice>(this), desc);
  }

  void SYCLDevice::imageCopy(const Image& src, const Image& dst)
  {
    xpuImageCopy(Ref<SYCLDevice>(this), src, dst);
  }

} // namespace oidn
