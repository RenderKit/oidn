// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include "metal_autoexposure.h"
#include "metal_common.h"

OIDN_NAMESPACE_BEGIN

  MetalAutoexposure::MetalAutoexposure(const Ref<MetalEngine>& engine, const ImageDesc& srcDesc)
    : Autoexposure(srcDesc),
      engine(engine),
      result(0),
      pipelineDownsample(nullptr), pipelineReduce(nullptr),
      commandQueue(nullptr), paramsBuffer(nullptr) {}

  MetalAutoexposure::~MetalAutoexposure()
  {
    cleanup();
  }

  void MetalAutoexposure::cleanup()
  {
    if (pipelineDownsample)
      [pipelineDownsample release];
    if (pipelineReduce)
      [pipelineReduce release];
    if (commandQueue)
      [commandQueue release];
    if (paramsBuffer)
      [paramsBuffer release];

    pipelineDownsample = nullptr;
    pipelineReduce = nullptr;
    commandQueue = nullptr;
    paramsBuffer = nullptr;
  }

  void MetalAutoexposure::finalize()
  {
    @autoreleasepool {
      MTLDevice_t device = static_cast<MetalDevice*>(engine->getDevice())->getMetalDevice();

      pipelineDownsample = createPipeline(device, "autoexposure_downsample");
      pipelineReduce = createPipeline(device, "autoexposure_reduce");

      commandQueue = [device newCommandQueue];

      if (!commandQueue)
        throw std::runtime_error("can not create command queue");

      paramsBuffer = [device newBufferWithLength: sizeof(ProcessParams)
                                         options: MTLResourceStorageModeShared];

      binsTensor = engine->newTensor({{numBins}, TensorLayout::x, DataType::Float32});
      sumsTensor = engine->newTensor({{numBins / maxBinSize}, TensorLayout::x, DataType::Float32});
      countsTensor = engine->newTensor({{numBins / maxBinSize}, TensorLayout::x, DataType::Float32});
    }
  }

  void MetalAutoexposure::submit()
  {
    AutoexposureParams params = createProcessParams();

    void* paramsPtr = [paramsBuffer contents];
    memcpy(paramsPtr, &params, sizeof(params));

    downsample();
    reduce();
  }

  void MetalAutoexposure::downsample()
  {
    auto width = maxBinSize;
    auto height = maxBinSize;

    id<MTLBuffer> input = getMTLBuffer(src->getBuffer());
    id<MTLBuffer> bins = getMTLBuffer(binsTensor->getBuffer());

    auto commandBuffer = [commandQueue commandBuffer];
    auto computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState: pipelineDownsample];

    [computeEncoder setThreadgroupMemoryLength: sizeof(float) * width * height
                                       atIndex: 0];

    int index = 0;

    [computeEncoder setBuffer: input
                       offset: 0
                      atIndex: index++];

    [computeEncoder setBuffer: bins
                       offset: 0
                      atIndex: index++];

    [computeEncoder setBuffer: paramsBuffer
                       offset: 0
                      atIndex: index++];

    auto threadsPerThreadgroup = MTLSizeMake(width, height, 1);

    auto threadgroupsPerGrid = MTLSizeMake((src->getW() + width - 1) / width,
                                           (src->getH() + height - 1) / height, 1);

    [computeEncoder dispatchThreadgroups: threadgroupsPerGrid
                   threadsPerThreadgroup: threadsPerThreadgroup];

    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    [computeEncoder release];
    [commandBuffer release];
  }

  void MetalAutoexposure::reduce()
  {
    auto width = [pipelineReduce threadExecutionWidth];
    auto height = 1;

    id<MTLBuffer> input = getMTLBuffer(binsTensor->getBuffer());
    id<MTLBuffer> sums = getMTLBuffer(sumsTensor->getBuffer());
    id<MTLBuffer> counts = getMTLBuffer(countsTensor->getBuffer());

    auto commandBuffer = [commandQueue commandBuffer];
    auto computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState: pipelineReduce];

    int index = 0;

    [computeEncoder setBuffer: input
                       offset: 0
                      atIndex: index++];

    [computeEncoder setBuffer: sums
                       offset: 0
                      atIndex: index++];

    [computeEncoder setBuffer: counts
                       offset: 0
                      atIndex: index++];

    [computeEncoder setBuffer: paramsBuffer
                       offset: 0
                      atIndex: index++];

    auto threadsPerThreadgroup = MTLSizeMake(width, height, 1);

    auto threadgroupsPerGrid = MTLSizeMake(width, 1, 1);

    [computeEncoder dispatchThreadgroups: threadgroupsPerGrid
                   threadsPerThreadgroup: threadsPerThreadgroup];

    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    float sum = ((float*)[sums contents])[0];
    float count = ((float*)[counts contents])[0];

    result = (count > 0) ? (Autoexposure::key / math::exp2(sum / float(count))) : 1.f;
  }

  AutoexposureParams MetalAutoexposure::createProcessParams()
  {
    AutoexposureParams params = {0};
    params.H = src->getH();
    params.W = src->getW();
    params.maxBinSize = maxBinSize;
    params.numBinsH = numBinsH;
    params.numBinsW = numBinsW;
    params.inputDataType = toDataType(src->getDataType());
    return params;
  }

OIDN_NAMESPACE_END
