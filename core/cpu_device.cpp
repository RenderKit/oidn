// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_device.h"
#include "cpu_buffer.h"

namespace oidn {

  void CPUDevice::init()
  {
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
  }

  void CPUDevice::printInfo()
  {
    std::cout << "  ISA     : ";
  #if defined(OIDN_X64)
    if (isISASupported(ISA::AVX512_CORE))
      std::cout << "AVX512";
    else if (isISASupported(ISA::AVX2))
      std::cout << "AVX2";
    else if (isISASupported(ISA::SSE41))
      std::cout << "SSE4.1";
  #elif defined(OIDN_ARM64)
    std::cout << "NEON";
  #endif
    std::cout << std::endl;
    
    std::cout << "  Neural  : ";
  #if defined(OIDN_DNNL)
    std::cout << "DNNL (oneDNN) " << DNNL_VERSION_MAJOR << "." <<
                                     DNNL_VERSION_MINOR << "." <<
                                     DNNL_VERSION_PATCH;
  #elif defined(OIDN_BNNS)
    std::cout << "BNNS";
  #endif
    std::cout << std::endl;
  }

  Ref<Buffer> CPUDevice::newBuffer(size_t byteSize, Buffer::Kind kind)
  {
    return makeRef<CPUBuffer>(Ref<Device>(this), byteSize);
  }

  Ref<Buffer> CPUDevice::newBuffer(void* ptr, size_t byteSize)
  {
    return makeRef<CPUBuffer>(Ref<Device>(this), ptr, byteSize);
  }

} // namespace oidn
