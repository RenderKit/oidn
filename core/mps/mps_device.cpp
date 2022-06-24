// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mps_device.h"
#include "../cpu_buffer.h"

namespace oidn {

  void MPSDevice::init()
  {
    // Initialize TBB
    initTasking();

#if defined(OIDN_DNNL)
    dnnl_set_verbose(clamp(verbose - 2, 0, 2)); // unfortunately this is not per-device but global
    dnnlEngine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    dnnlStream = dnnl::stream(dnnlEngine);
#endif
    tensorBlockSize = 1;
  }

  void MPSDevice::printInfo()
  {
  }

  Ref<Buffer> MPSDevice::newBuffer(size_t byteSize, Buffer::Kind kind)
  {
    return makeRef<CPUBuffer>(Ref<Device>(this), byteSize);
  }

  Ref<Buffer> MPSDevice::newBuffer(void* ptr, size_t byteSize)
  {
    return makeRef<CPUBuffer>(Ref<Device>(this), ptr, byteSize);
  }

} // namespace oidn
