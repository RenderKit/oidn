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
    std::cout << "  Targets :";
  #if defined(OIDN_X64)
    if (isISASupported(ISA::SSE41))       std::cout << " SSE4.1";
    if (isISASupported(ISA::AVX2))        std::cout << " AVX2";
    if (isISASupported(ISA::AVX512_CORE)) std::cout << " AVX512";
  #elif defined(OIDN_ARM64)
    std::cout << " NEON";
  #endif
    std::cout << " (supported)" << std::endl;
    std::cout << "            ";
  #if defined(OIDN_X64)
    std::cout << "SSE4.1 AVX2 AVX512";
  #elif defined(OIDN_ARM64)
    std::cout << "NEON";
  #endif
    std::cout << " (compile time enabled)" << std::endl;
    
    std::cout << "  Neural  : ";
  #if defined(OIDN_DNNL)
    std::cout << "DNNL (oneDNN) " << DNNL_VERSION_MAJOR << "." <<
                                     DNNL_VERSION_MINOR << "." <<
                                     DNNL_VERSION_PATCH;
  #elif defined(OIDN_BNNS)
    std::cout << "BNNS";
  #endif
    std::cout << std::endl;

    std::cout << "  Tasking :";
    std::cout << " TBB" << TBB_VERSION_MAJOR << "." << TBB_VERSION_MINOR;
  #if TBB_INTERFACE_VERSION >= 12002
    std::cout << " TBB_header_interface_" << TBB_INTERFACE_VERSION << " TBB_lib_interface_" << TBB_runtime_interface_version();
  #else
    std::cout << " TBB_header_interface_" << TBB_INTERFACE_VERSION << " TBB_lib_interface_" << tbb::TBB_runtime_interface_version();
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
