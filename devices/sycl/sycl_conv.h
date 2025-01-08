// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/conv.h"
#include "sycl_engine.h"

OIDN_NAMESPACE_BEGIN

  namespace xelp {
    Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc);
  }

  namespace xehpg {
    Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc);
  }

#if defined(__linux__)
  namespace xehpc {
    Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc);
  }
#endif

  namespace xe2 {
    Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc);
  }

OIDN_NAMESPACE_END
