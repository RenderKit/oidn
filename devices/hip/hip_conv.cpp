// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_conv.h"

OIDN_NAMESPACE_BEGIN

  HIPConv::HIPConv(const Ref<HIPEngine>& engine, const ConvDesc& desc)
    : Conv(desc),
      engine(engine)
  {
    checkError(miopenCreateConvolutionDescriptor(&convDesc));
    checkError(miopenInitConvolutionDescriptor(
      convDesc,
      miopenConvolution, 
      1,
      1,
      1,
      1,
      1,
      1));

    checkError(miopenCreateActivationDescriptor(&activationDesc));
    checkError(miopenSetActivationDescriptor(
      activationDesc,
      activation == Activation::ReLU ? miopenActivationRELU : miopenActivationPASTHRU,
      0, 0, 0));

    xDesc = toMIOpen(srcDesc);
    wDesc = toMIOpen(weightDesc);
    bDesc = toMIOpen(biasDesc);
    yDesc = toMIOpen(dstDesc);
  }

  HIPConv::~HIPConv()
  {
    checkError(miopenDestroyConvolutionDescriptor(convDesc));
    checkError(miopenDestroyActivationDescriptor(activationDesc));
    checkError(miopenDestroyTensorDescriptor(xDesc));
    checkError(miopenDestroyTensorDescriptor(wDesc));
    checkError(miopenDestroyTensorDescriptor(bDesc));
    checkError(miopenDestroyTensorDescriptor(yDesc));
  }

  bool HIPConv::isSupported() const
  {
    return xDesc && wDesc && bDesc && yDesc;
  }

  size_t HIPConv::getScratchByteSize() const
  {
    assert(isSupported());
    
    size_t scratchByteSize;
    checkError(miopenConvolutionForwardGetWorkSpaceSize(engine->getMIOpenHandle(),
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
    if (finalized)
      throw std::logic_error("convolution already finalized");
    if (!src || !weight || !dst)
      throw std::logic_error("convolution source/weight/destination not set before finalization");

    int returnedAlgoCount;
    miopenConvAlgoPerf_t perfResults;

    checkError(miopenFindConvolutionForwardAlgorithm(engine->getMIOpenHandle(),
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
    finalized = true;
  }

  void HIPConv::submit()
  {
    if (!finalized)
      throw std::logic_error("convolution not finalized");

    const float alpha = 1;
    const float beta = 0;

    checkError(miopenConvolutionForward(engine->getMIOpenHandle(),
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
    //miopenGetKernelTime(engine->getMIOpenHandle(), &time);
    //std::cout << "conv " << weight->getO() << "x" << weight->getI() << "x" << src->getH() << "x" << src->getW() << ": " << time << std::endl;;

    checkError(miopenConvolutionForwardBias(engine->getMIOpenHandle(),
                                            &alpha,
                                            bDesc,
                                            bias->getData(),
                                            &beta,
                                            yDesc,
                                            dst->getData()));

    if (activation != Activation::None)
    {
      checkError(miopenActivationForward(engine->getMIOpenHandle(),
                                         activationDesc,
                                         &alpha,
                                         yDesc,
                                         dst->getData(),
                                         &beta,
                                         yDesc,
                                         dst->getData()));
    }
  }

OIDN_NAMESPACE_END