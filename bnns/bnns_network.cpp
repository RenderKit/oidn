#include "bnns_network.h"
#include "bnns_node.h"
#include "../core/color.h"

#if defined(USE_BNNS)

namespace oidn
{
std::shared_ptr<Node> BnnsNetwork::addConv(const std::string& name,
                              const std::shared_ptr<memory>& src,
                              const std::shared_ptr<memory>& userDst,
                              bool relu)
{

    const auto& W = weightsMap[name + ".weight"];
    if (W.ndims() != 4 || W.layout != "oihw")
      throw Exception(Error::InvalidOperation, "invalid convolution weights");
    auto weights = padWeights(W);
    memory::dims weightsDims = getMemoryDims(weights);

    // Get and pad the biases
    const auto& b = weightsMap[name + ".bias"];
    if (b.ndims() != 1)
      throw Exception(Error::InvalidOperation, "invalid convolution biases");
    auto bias = padBias(b);
    memory::dims biasDims = getMemoryDims(bias);

    memory::dims srcDims = getMemoryDims(src);
    memory::dims dstDims = srcDims;
    dstDims[1] = weightsDims[0]; // dstDims[C] = weightsDims[OC]

    std::shared_ptr<memory> dst;
    if (!userDst)
      dst = allocMemory(dstDims);
    else if (getMemoryDims(userDst) == dstDims)
      dst = userDst;
    else
      dst = castMemory(dstDims, userDst);

    BNNSLayerParametersConvolution out = (BNNSLayerParametersConvolution)
    {
      .x_stride=1,
      .y_stride = 1,
      .x_dilation_stride=1,
      .y_dilation_stride=1,
      .x_padding = 1,
      .y_padding = 1,
    };

    out.i_desc = (BNNSNDArrayDescriptor){.layout = BNNSDataLayoutImageCHW,
                                                  .size = {(size_t)srcDims[3], (size_t)srcDims[2], (size_t)srcDims[1]},
                                                  .stride = {0, 0, 0},
                                                  .data = NULL, //input data pointer can be NULL during filter create
                                                  .data_type = BNNSDataTypeFloat32};
    out.w_desc = (BNNSNDArrayDescriptor){.layout = BNNSDataLayoutConvolutionWeightsOIHW,
                                                  .size = {3, 3, (size_t)srcDims[1] , (size_t)dstDims[1]},
                                                  .stride = {0, 0, 0, 0},
                                                  .data = weights->map_data(),
                                                  .data_type = BNNSDataTypeFloat32};
    out.bias = (BNNSNDArrayDescriptor){.layout = BNNSDataLayoutVector,
                                                .size = {(size_t)dstDims[1]},
                                                .stride = {0},
                                                .data = bias->map_data(),
                                                .data_type = BNNSDataTypeFloat32};
    out.o_desc = (BNNSNDArrayDescriptor){.layout = BNNSDataLayoutImageCHW,
                                                  .size = {(size_t)dstDims[3],(size_t)dstDims[2],(size_t)dstDims[1]},
                                                  .stride = {0, 0, 0},
                                                  .data = NULL,  //output data pointer can be NULL during filter create
                                                  .data_type = BNNSDataTypeFloat32};
    out.activation.function = BNNSActivationFunctionRectifiedLinear;

    if (!relu)
        out.activation.function = BNNSActivationFunctionIdentity;

    std::shared_ptr<Node> node = std::make_shared<BnnsConvNode>(out,src,dst);
    nodes.push_back(node);
    return node;
}


std::shared_ptr<Node> BnnsNetwork::addPool(const std::shared_ptr<memory>& src,
                                           const std::shared_ptr<memory>& userDst)
{
    memory::dims srcDims = getMemoryDims(src);
    memory::dims dstDims = getPoolDims(srcDims);

    std::shared_ptr<memory> dst;
    if (!userDst)
      dst = allocMemory(dstDims);
    else if (getMemoryDims(userDst) == dstDims)
      dst = userDst;
    else
      dst = castMemory(dstDims, userDst);

    BNNSLayerParametersPooling desc = (BNNSLayerParametersPooling){.pooling_function = BNNSPoolingFunctionMax,
                                                .k_width = 2,
                                                .k_height = 2,
                                                .x_stride = 2,
                                                .y_stride = 2};

    desc.i_desc = (BNNSNDArrayDescriptor){.layout = BNNSDataLayoutImageCHW,
        .size = {(size_t)srcDims[3], (size_t)srcDims[2], (size_t)srcDims[1]},
        .stride = {0, 0, 0},
        .data = NULL,
        .data_type = BNNSDataTypeFloat32};

    desc.o_desc = (BNNSNDArrayDescriptor){.layout = BNNSDataLayoutImageCHW,
                                                  .size = {(size_t)srcDims[3]/2, (size_t)srcDims[2]/2, (size_t)srcDims[1]},
                                                  .stride = {0, 0, 0},
                                                  .data = NULL,
                                                  .data_type = BNNSDataTypeFloat32};


    std::shared_ptr<Node> node = std::make_shared<BnnsPoolNode>(desc,src,dst);
    nodes.push_back(node);
    return node;
}


std::shared_ptr<Node> BnnsNetwork::addUpsample(const std::shared_ptr<memory>& src,
                                               const std::shared_ptr<memory>& userDst)
{
  memory::dims srcDims = getMemoryDims(src);
  memory::dims dstDims = getUpsampleDims(srcDims);

  std::shared_ptr<memory> dst;
  if (!userDst)
    dst = allocMemory(dstDims);
  else if (getMemoryDims(userDst) == dstDims)
    dst = userDst;
  else
    dst = castMemory(dstDims, userDst);

  auto node = std::make_shared<BnnsUpsampleNode>(src, dst);
  nodes.push_back(node);
  return node;
}



std::shared_ptr<Node> BnnsNetwork::addInputReorder(const Image& color,
                                                   const Image& albedo,
                                                   const Image& normal,
                                                   const std::shared_ptr<TransferFunction>& transferFunc,
                                                   bool hdr,
                                                   int alignment,
                                                   const std::shared_ptr<memory>& userDst)
{
  assert(color);
  int inputC = 3;
  if (albedo) inputC += 3;
  if (normal) inputC += 3;

  memory::dims srcDims = {1, inputC, color.height, color.width};
  memory::dims dstDims = getInputReorderDims(srcDims, alignment);

  // Allocate padded memory
  auto dst = userDst;
  if (!dst)
    dst = allocMemory(dstDims);

  auto node = std::make_shared<BnnsInputReorderNode>(color, albedo, normal, dst, transferFunc, hdr);
  nodes.push_back(node);
  return node;
}


std::shared_ptr<Node> BnnsNetwork::addOutputReorder(const std::shared_ptr<memory>& src,
                                                    const std::shared_ptr<TransferFunction>& transferFunc,
                                                    bool hdr,
                                                    const Image& output)
{
  memory::dims srcDims = getMemoryDims(src);
  assert(srcDims[1] == K);

  // Push node
  auto node = std::make_shared<BnnsOutputReorderNode>(src, output, transferFunc, hdr);
  nodes.push_back(node);
  return node;
}



void BnnsNetwork::execute(Progress& progress)
{
  for (size_t i = 0; i < nodes.size(); ++i)
  {
    nodes[i]->execute(NULL);
    progress.update(1);
  }
}



};

#endif



