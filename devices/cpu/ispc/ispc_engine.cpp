#include "ispc_engine.h"
#include "ispc_conv.h"

OIDN_NAMESPACE_BEGIN

  ISPCEngine::ISPCEngine(const Ref<CPUDevice>& device)
    : CPUEngine(device)
    {}

  std::shared_ptr<Conv> ISPCEngine::newConv(const ConvDesc& desc)
  {
    return std::make_shared<ISPCConv>(this, desc);
  }

OIDN_NAMESPACE_END