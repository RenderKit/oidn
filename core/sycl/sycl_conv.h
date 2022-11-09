// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "sycl_engine.h"

namespace oidn {

  namespace xehpg { std::shared_ptr<Conv> newConv(const Ref<SYCLEngine>& engine, const ConvDesc& desc); }
  namespace xehpc { std::shared_ptr<Conv> newConv(const Ref<SYCLEngine>& engine, const ConvDesc& desc); }
  namespace gen9  { std::shared_ptr<Conv> newConv(const Ref<SYCLEngine>& engine, const ConvDesc& desc); }

} // namespace oidn
