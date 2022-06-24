// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "network.h"
#include "mps/MPSGraphNetwork-C-Interface.h"

namespace oidn {

  class MPSNetwork : public Network
  {
  public:
    MPSNetwork(const Ref<Device>& device, const std::map<std::string, std::shared_ptr<Tensor>>& weightsMap);
    virtual ~MPSNetwork() = default;

    void execute(Progress& progress) override;
    
    double getWorkAmount() const override;
    
    std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, ptrdiff_t offset) override;
    
    std::shared_ptr<InputReorderNode> addInputReorder(const std::string& name,
                                                      const std::shared_ptr<Tensor>& dst,
                                                      const std::shared_ptr<TransferFunction>& transferFunc,
                                                      bool hdr,
                                                      bool snorm) override;
    
    std::shared_ptr<OutputReorderNode> addOutputReorder(const std::string& name,
                                                        const std::shared_ptr<Tensor>& src,
                                                        const std::shared_ptr<TransferFunction>& transferFunc,
                                                        bool hdr,
                                                        bool snorm) override;
    
    std::shared_ptr<Node> addConv(const std::string& name,
                                  const std::shared_ptr<Tensor>& src,
                                  const std::shared_ptr<Tensor>& dst,
                                  bool relu = true) override;

    std::shared_ptr<Node> addPool(const std::string& name,
                                  const std::shared_ptr<Tensor>& src,
                                  const std::shared_ptr<Tensor>& dst) override;

    std::shared_ptr<Node> addUpsample(const std::string& name,
                                      const std::shared_ptr<Tensor>& src,
                                      const std::shared_ptr<Tensor>& dst) override;
    
    std::shared_ptr<Node> addConcat(const std::string& name,
                                    const std::shared_ptr<Tensor>& src1,
                                    const std::shared_ptr<Tensor>& src2,
                                    const std::shared_ptr<Tensor>& dst) override;
    
  private:
    std::unique_ptr<MPSGraphNetworkImpl> mpsNet;
    std::map<std::shared_ptr<Tensor>, std::string> tensorToName;
    
    std::shared_ptr<InputReorderNode> inputReorder;
    std::shared_ptr<OutputReorderNode> outputReorder;
  };

} // namespace oidn
