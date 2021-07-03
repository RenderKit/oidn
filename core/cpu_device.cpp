// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_device.h"
#include "cpu_buffer.h"

namespace oidn {

  void CPUDevice::commit()
  {
    if (isCommitted())
      throw Exception(Error::InvalidOperation, "device can be committed only once");

    // Initialize TBB
    initTasking();

    // Initialize the neural network runtime
  #if defined(OIDN_DNNL)
    dnnl_set_verbose(clamp(verbose - 2, 0, 2)); // unfortunately this is not per-device but global
    dnnlEngine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    dnnlStream = dnnl::stream(dnnlEngine);
    tensorBlockSize = isISASupported(ISA::AVX512_CORE) ? 16 : 8;
  #else
    tensorBlockSize = 1;
  #endif

    dirty = false;
    committed = true;

    if (isVerbose())
      print();
  }

  Ref<Buffer> CPUDevice::newBuffer(size_t byteSize)
  {
    return makeRef<CPUBuffer>(Ref<Device>(this), byteSize);
  }

  Ref<Buffer> CPUDevice::newBuffer(void* ptr, size_t byteSize)
  {
    return makeRef<CPUBuffer>(Ref<Device>(this), ptr, byteSize);
  }

} // namespace oidn
