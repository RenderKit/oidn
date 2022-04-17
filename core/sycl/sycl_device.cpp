// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mkl-dnn/include/dnnl_sycl.hpp"
#include "../gpu/gpu_autoexposure.h"
#include "../gpu/gpu_input_process.h"
#include "../gpu/gpu_output_process.h"
#include "../gpu/gpu_image_copy.h"
#include "sycl_device.h"
#include "sycl_op.h"
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

    if (isVerbose())
      std::cout << "  Device  : " << sycl->device.get_info<sycl::info::device::name>() << std::endl;

    // Check required hardware features
    if (!sycl->device.has(sycl::aspect::usm_host_allocations) ||
        !sycl->device.has(sycl::aspect::usm_device_allocations) ||
        !sycl->device.has(sycl::aspect::usm_shared_allocations))
      throw Exception(Error::UnsupportedHardware, "device does not support unified shared memory");

    tensorDataType = DataType::Float16;
    tensorLayout = TensorLayout::Chw16c;
    tensorBlockSize = 16;

    if (isVerbose())
    {
      std::cout << "  Neural  : ";
      std::cout << "DNNL (oneDNN) " << DNNL_VERSION_MAJOR << "." <<
                                       DNNL_VERSION_MINOR << "." <<
                                       DNNL_VERSION_PATCH;
      std::cout << std::endl;
    }
  }

  std::shared_ptr<Pool> SYCLDevice::newPool(const PoolDesc& desc)
  {
    return std::make_shared<SYCLPool>(this, desc);
  }

  std::shared_ptr<Upsample> SYCLDevice::newUpsample(const UpsampleDesc& desc)
  {
    return std::make_shared<SYCLUpsample>(this, desc);
  }

  std::shared_ptr<Autoexposure> SYCLDevice::newAutoexposure(const ImageDesc& srcDesc)
  {
    return std::make_shared<GPUAutoexposure<SYCLOp>>(this, srcDesc);
  }

  std::shared_ptr<InputProcess> SYCLDevice::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<GPUInputProcess<SYCLOp, half, TensorLayout::Chw16c>>(this, desc);
  }

  std::shared_ptr<OutputProcess> SYCLDevice::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<GPUOutputProcess<SYCLOp, half, TensorLayout::Chw16c>>(this, desc);
  }

  std::shared_ptr<ImageCopy> SYCLDevice::newImageCopy()
  {
    return std::make_shared<GPUImageCopy<SYCLOp>>(this);
  }

  void* SYCLDevice::malloc(size_t byteSize, Storage storage)
  {
    switch (storage)
    {
    case Storage::Undefined:
    case Storage::Host:
      return sycl::aligned_alloc_host(memoryAlignment,
                                      byteSize,
                                      sycl->context);

    case Storage::Device:
      return sycl::aligned_alloc_device(memoryAlignment,
                                        byteSize,
                                        sycl->device,
                                        sycl->context);

    case Storage::Managed:
      return sycl::aligned_alloc_shared(memoryAlignment,
                                        byteSize,
                                        sycl->device,
                                        sycl->context);

    default:
      throw Exception(Error::InvalidArgument, "invalid storage mode");
    }
  }

  void SYCLDevice::free(void* ptr, Storage storage)
  {
    sycl::free(ptr, sycl->context);
  }

  void SYCLDevice::memcpy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    sycl->queue.memcpy(dstPtr, srcPtr, byteSize).wait();
  }

} // namespace oidn
