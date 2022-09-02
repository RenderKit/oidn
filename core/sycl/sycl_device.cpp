// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_device.h"
#include "../gpu/gpu_autoexposure.h"
#include "../gpu/gpu_input_process.h"
#include "../gpu/gpu_output_process.h"
#include "../gpu/gpu_image_copy.h"
#include "sycl_conv_mad.h"
#include "sycl_conv_dpas.h"
#include "sycl_conv_pvc.h"
#include "sycl_pool.h"
#include "sycl_upsample.h"

namespace oidn {

  class SYCLDeviceSelector : public sycl::device_selector
  {
  public:
    int operator()(const sycl::device& device) const override
    {
      if (!SYCLDevice::isDeviceSupported(device))
        return -1;

      // FIXME: improve detection of fastest discrete GPU
      return device.get_info<sycl::info::device::max_compute_units>() * device.get_info<sycl::info::device::max_work_group_size>();
    }
  };

  bool SYCLDevice::isSupported()
  {
    for (const auto& platform : sycl::platform::get_platforms())
      for (const auto& device : platform.get_devices(sycl::info::device_type::gpu))
        if (isDeviceSupported(device))
          return true;
    return false;
  }

  bool SYCLDevice::isDeviceSupported(const sycl::device& device)
  {
    return device.is_gpu() &&
           device.get_info<sycl::info::device::vendor_id>() == 0x8086 && // Intel
           device.has(sycl::aspect::usm_host_allocations) &&
           device.has(sycl::aspect::usm_device_allocations) &&
           device.has(sycl::aspect::usm_shared_allocations);
  }

  SYCLDevice::SYCLDevice() {}

  SYCLDevice::SYCLDevice(const sycl::queue& syclQueue)
    : sycl(new SYCL{syclQueue.get_context(), syclQueue.get_device(), syclQueue}) {}

  void SYCLDevice::init()
  {
    if (!sycl)
    {
      // Initialize the SYCL device and queue
      sycl::queue syclQueue(SYCLDeviceSelector(),
                            sycl::property_list{sycl::property::queue::in_order{}});

      sycl.reset(new SYCL{syclQueue.get_context(), syclQueue.get_device(), syclQueue});
    }

    if (isVerbose())
    {
      std::cout << "  Device    : " << sycl->device.get_info<sycl::info::device::name>() << std::endl;
      std::cout << "    EUs     : " << sycl->device.get_info<sycl::info::device::max_compute_units>() << std::endl;
      std::cout << "    Platform: " << sycl->device.get_platform().get_info<sycl::info::platform::name>() << std::endl;
    }

    // Check the SYCL device and queue
    if (!isDeviceSupported(sycl->device))
      throw Exception(Error::UnsupportedHardware, "unsupported SYCL device");
    if (!sycl->queue.is_in_order())
        throw Exception(Error::InvalidArgument, "unsupported out-of-order SYCL queue");

    maxWorkGroupSize = sycl->device.get_info<sycl::info::device::max_work_group_size>();

    tensorDataType  = DataType::Float16;
    tensorLayout    = TensorLayout::Chw16c;
    //weightsLayout   = TensorLayout::OIhw2o8i8o2i;
    weightsLayout   = TensorLayout::OIhw8i16o2i;
    tensorBlockSize = 16;
  }

  std::shared_ptr<Conv> SYCLDevice::newConv(const ConvDesc& desc)
  {
    //return std::make_shared<SYCLConvDPAS>(this, desc);
    return std::make_shared<SYCLConvPVC>(this, desc);
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
    if (maxWorkGroupSize >= 1024)
      return std::make_shared<GPUAutoexposure<SYCLDevice, 1024>>(this, srcDesc);
    else if (maxWorkGroupSize >= 512)
      return std::make_shared<GPUAutoexposure<SYCLDevice, 512>>(this, srcDesc);
    else
      return std::make_shared<GPUAutoexposure<SYCLDevice, 256>>(this, srcDesc);
  }

  std::shared_ptr<InputProcess> SYCLDevice::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<GPUInputProcess<SYCLDevice, half, TensorLayout::Chw16c>>(this, desc);
  }

  std::shared_ptr<OutputProcess> SYCLDevice::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<GPUOutputProcess<SYCLDevice, half, TensorLayout::Chw16c>>(this, desc);
  }

  std::shared_ptr<ImageCopy> SYCLDevice::newImageCopy()
  {
    return std::make_shared<GPUImageCopy<SYCLDevice>>(this);
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

  Storage SYCLDevice::getPointerStorage(const void* ptr)
  {
    switch (sycl::get_pointer_type(ptr, sycl->context))
    {
      case sycl::usm::alloc::host:
        return Storage::Host;

      case sycl::usm::alloc::device:
        return sycl::get_pointer_device(ptr, sycl->context) == sycl->device ? Storage::Device : Storage::Undefined;
      
      case sycl::usm::alloc::shared:
        return Storage::Managed;
      
      default:
        return Storage::Undefined;
    }
  }

  void SYCLDevice::runHostFuncAsync(std::function<void()>&& f)
  {
    sycl->queue.submit([&](sycl::handler& cgh) { cgh.host_task(f); });
  }

  void SYCLDevice::wait()
  {
    sycl->queue.wait_and_throw();
  }

} // namespace oidn
