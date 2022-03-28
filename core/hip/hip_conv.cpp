// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_conv.h"

namespace oidn {

  HIPConv::HIPConv(const Ref<HIPDevice>& device, const ConvDesc& desc)
    : HIPOp(device),
      Conv(desc)
  {
    checkError(miopenCreateConvolutionDescriptor(&convDesc));
    checkError(miopenInitConvolutionDescriptor(convDesc,
                                               miopenConvolution, 
                                               1,
                                               1,
                                               1,
                                               1,
                                               1,
                                               1));

    checkError(miopenCreateActivationDescriptor(&activationDesc));
    checkError(miopenSetActivationDescriptor(activationDesc,
                                             relu ? miopenActivationRELU : miopenActivationPASTHRU,
                                             0, 0, 0));

    xDesc    = toMIOpen(srcDesc);
    wDesc    = toMIOpen(weight->getDesc());
    biasDesc = toMIOpen(bias->getDesc());
    yDesc    = toMIOpen(dstDesc);
  }

  HIPConv::~HIPConv()
  {
    checkError(miopenDestroyConvolutionDescriptor(convDesc));
    checkError(miopenDestroyActivationDescriptor(activationDesc));
    checkError(miopenDestroyTensorDescriptor(xDesc));
    checkError(miopenDestroyTensorDescriptor(wDesc));
    checkError(miopenDestroyTensorDescriptor(biasDesc));
    checkError(miopenDestroyTensorDescriptor(yDesc));
  }

  bool HIPConv::isSupported() const
  {
    return xDesc && wDesc && biasDesc && yDesc;
  }

  size_t HIPConv::getScratchByteSize() const
  {
    assert(isSupported());
    
    size_t scratchByteSize;
    checkError(miopenConvolutionForwardGetWorkSpaceSize(device->getMIOpenHandle(),
                                                        wDesc,
                                                        xDesc,
                                                        convDesc,
                                                        yDesc,
                                                        &scratchByteSize));
    return scratchByteSize;
  }

  void HIPConv::setScratch(const std::shared_ptr<Tensor>& scratch)
  {
    this->scratch = scratch;
  }

  void HIPConv::finalize()
  {
    int returnedAlgoCount;
    miopenConvAlgoPerf_t perfResults;

    checkError(miopenFindConvolutionForwardAlgorithm(device->getMIOpenHandle(),
                                                     xDesc,
                                                     src->getData(),
                                                     wDesc,
                                                     weight->getData(),
                                                     convDesc,
                                                     yDesc,
                                                     dst->getData(),
                                                     1,
                                                     &returnedAlgoCount,
                                                     &perfResults,
                                                     scratch ? scratch->getData() : nullptr,
                                                     scratch ? scratch->getByteSize() : 0,
                                                     true));

    algo = perfResults.fwd_algo;
    //std::cout << "ALGO found: " << int(algo) << " " << perfResults.memory << std::endl;
  }

  void HIPConv::run()
  {
    assert(isSupported());

    const float alpha = 1;
    const float beta = 0;

    checkError(miopenConvolutionForward(device->getMIOpenHandle(),
                                        &alpha,
                                        xDesc,
                                        src->getData(),
                                        wDesc,
                                        weight->getData(),
                                        convDesc,
                                        algo,
                                        &beta,
                                        yDesc,
                                        dst->getData(),
                                        scratch ? scratch->getData() : nullptr,
                                        scratch ? scratch->getByteSize() : 0));

    //float time;
    //miopenGetKernelTime(device->getMIOpenHandle(), &time);
    //std::cout << "conv " << weight->getO() << "x" << weight->getI() << "x" << src->getH() << "x" << src->getW() << ": " << time << std::endl;;

    checkError(miopenConvolutionForwardBias(device->getMIOpenHandle(),
                                            &alpha,
                                            biasDesc,
                                            bias->getData(),
                                            &beta,
                                            yDesc,
                                            dst->getData()));

    if (relu)
    {
      checkError(miopenActivationForward(device->getMIOpenHandle(),
                                         activationDesc,
                                         &alpha,
                                         yDesc,
                                         dst->getData(),
                                         &beta,
                                         yDesc,
                                         dst->getData()));
    }
  }

} // namespace oidn