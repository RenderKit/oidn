// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sycl_device.h"

OIDN_NAMESPACE_BEGIN

  // Table of supported architectures and corresponding IP versions with revisions masked out
  // These should match the AOT targets defined in CMakeLists.txt
  // https://github.com/intel/compute-runtime/blob/14251c3d96e71e97e397b0c4fcb01557fca47f0e/shared/source/helpers/hw_ip_version.h
  // https://github.com/intel/compute-runtime/blob/master/third_party/aot_config_headers/platforms.h
  struct SYCLDeviceTableEntry
  {
    SYCLArch arch;
    std::vector<uint32_t> ipVersions;
  };

  constexpr uint32_t syclDeviceIPVersionMask = 0xffffffc0;

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
        0x03118000, // mtl-u
        0x0311c000, // mtl-h
      }
    },
    {
      SYCLArch::XeLPGplus,
      {
        0x03128000, // arl-h
      }
    },
    {
      SYCLArch::XeHPG,
      {
        0x030dc000, // acm-g10
        0x030e0000, // acm-g11
        0x030e4000, // acm-g12
      }
    },
    #if defined(__linux__)
    {
      SYCLArch::XeHPC,
      {
        0x030f0000, // pvc-sdv, pvc
      }
    },
    {
      SYCLArch::XeHPC_NoDPAS,
      {
        0x030f4000, // pvc-vg
      }
    },
    #endif
    {
      SYCLArch::Xe2LPG,
      {
        0x05010000, // lnl-m
      }
    },
    {
      SYCLArch::Xe2HPG,
      {
        0x05004000, // bmg-g21
      }
    },
    {
      SYCLArch::Xe3LPG,
      {
        0x07800000, // ptl-h
        0x07804000, // ptl-u
      }
    },
  };

OIDN_NAMESPACE_END