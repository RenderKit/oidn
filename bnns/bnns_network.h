
#include "../core/network.h"

#pragma once


#if defined(USE_BNNS)
namespace oidn {
class BnnsNetwork : public Network
{
protected:
    memory::format_tag getMemoryFormat() override { return memory::format_tag::nchw; }

public:
    BnnsNetwork(const Ref<Device>& device, const std::map<std::string, Tensor>& weightsMap)
                : Network(device, weightsMap)
                {}

    void execute(Progress& progress) override;


    std::shared_ptr<Node> addConv(const std::string& name,
                                  const std::shared_ptr<memory>& src,
                                  const std::shared_ptr<memory>& userDst = nullptr,
                                  bool relu = true) override;

    std::shared_ptr<Node> addPool(const std::shared_ptr<memory>& src,
                                  const std::shared_ptr<memory>& userDst = nullptr) override;

    std::shared_ptr<Node> addUpsample(const std::shared_ptr<memory>& src,
                                      const std::shared_ptr<memory>& userDst = nullptr) override;

    std::shared_ptr<Node> addInputReorder(const Image& color,
                                          const Image& albedo,
                                          const Image& normal,
                                          const std::shared_ptr<TransferFunction>& transferFunc,
                                          bool hdr,
                                          int alignment,
                                          const std::shared_ptr<memory>& userDst = nullptr) override;

    std::shared_ptr<Node> addOutputReorder(const std::shared_ptr<memory>& src,
                                           const std::shared_ptr<TransferFunction>& transferFunc,
                                           bool hdr,
                                           const Image& output) override;

};
};
#endif

