// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sycl_device.h"

OIDN_NAMESPACE_BEGIN

  // Table of supported device architectures and corresponding IP versions
  // These should match the AOT targets defined in CMakeLists.txt
  // https://github.com/intel/compute-runtime/blob/14251c3d96e71e97e397b0c4fcb01557fca47f0e/shared/source/helpers/hw_ip_version.h
  // https://github.com/intel/compute-runtime/blob/master/third_party/aot_config_headers/platforms.h
  struct SYCLDeviceTableEntry
  {
    SYCLArch arch;
    std::vector<uint32_t> ipVersions;
  };

  inline const std::vector<SYCLDeviceTableEntry> syclDeviceTable =
  {
    {
      SYCLArch::XeLP,
      {
        0x03000000, // tgllp
        0x03004000, // rkl
        0x03008000, // adl-s
        0x0300c000, // adl-p
        0x03010000, // adl-n
        0x03028000, // dg1
      }
    },
    {
      SYCLArch::XeLPG,
      {
        0x03118004, // mtl-m-b0
        0x0311c004, // mtl-p-b0
      }
    },
    {
      SYCLArch::XeHPG,
      {
        0x030dc008, // acm-g10-c0
        0x030e0005, // acm-g11-b1
        0x030e4000, // acm-g12-a0
      }
    },
    {
      SYCLArch::XeHPC,
      {
        0x030f0001, // pvc-xl-a0p (pvc-sdv)
        0x030f0007, // pvc-xt-c0  (pvc)
      }
    },
    #if !defined(OIDN_DEVICE_SYCL_AOT)
    {
      SYCLArch::XeHPC_NoDPAS,
      {
        0x030f4007, // pvc-xt-c0-vg
      }
    }
    #endif
  };

OIDN_NAMESPACE_END