#pragma once

#include "core/conv.h"
#include "ispc_engine.h"
#include "core/tensor.h"
#include "../cpu_common.h"

OIDN_NAMESPACE_BEGIN

  class ISPCConv : public Conv
  {
  public:
    ISPCConv(const Ref<ISPCEngine>& engine, const ConvDesc& desc);

  private:
    void submit() override;

    Ref<ISPCEngine> engine;
  };

OIDN_NAMESPACE_END