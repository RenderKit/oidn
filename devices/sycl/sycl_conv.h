// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/conv.h"
#include "sycl_engine.h"

OIDN_NAMESPACE_BEGIN

  namespace xelp {
    std::shared_ptr<Conv> newSYCLConv(const Ref<SYCLEngine>& engine, const ConvDesc& desc);
  }

  namespace xehpg {
    std::shared_ptr<Conv> newSYCLConv(const Ref<SYCLEngine>& engine, const ConvDesc& desc);
  }

#if defined(__linux__)
  namespace xehpc {
    std::shared_ptr<Conv> newSYCLConv(const Ref<SYCLEngine>& engine, const ConvDesc& desc);
  }
  namespace xehpc_fast {
    std::shared_ptr<Conv> newSYCLConv(const Ref<SYCLEngine>& engine, const ConvDesc& desc);
  }
#endif

OIDN_NAMESPACE_END
