// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_device.h"
#include "sycl_buffer.h"
#include "mkl-dnn/include/dnnl_sycl.hpp"

namespace oidn {

  void SYCLDevice::init()
  {
    // Initialize TBB (FIXME: remove)
    initTasking();

    // Initialize the neural network runtime
    dnnl_set_verbose(clamp(verbose - 2, 0, 2)); // unfortunately this is not per-device but global
    dnnlEngine  = dnnl::engine(dnnl::engine::kind::gpu, 0);
    dnnlStream  = dnnl::stream(dnnlEngine, dnnl::stream::flags::in_order);
    syclDevice  = dnnl::sycl_interop::get_device(dnnlEngine);
    syclContext = dnnl::sycl_interop::get_context(dnnlEngine);
    syclQueue   = dnnl::sycl_interop::get_queue(dnnlStream);

    tensorDataType = DataType::Float16;
    tensorBlockSize = 16;
  }

  void SYCLDevice::printInfo()
  {
    std::cout << "  Device  : " << syclDevice.get_info<sycl::info::device::name>() << std::endl;

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

} // namespace oidn
