// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include "metal_input_process.h"
#include "metal_engine.h"
#include "metal_buffer.h"
#include "metal_common.h"
#include "metal_kernel_common.h"

OIDN_NAMESPACE_BEGIN

  MetalInputProcess::MetalInputProcess(const Ref<MetalEngine>& engine, const InputProcessDesc& desc)
    : InputProcess(engine, desc),
      engine(engine),
      pipeline(nullptr), paramsBuffer(nullptr) {}

  MetalInputProcess::~MetalInputProcess()
  {
    cleanup();
  }

  void MetalInputProcess::cleanup()
  {
    if (pipeline)
      [pipeline release];
    if (paramsBuffer)
      [paramsBuffer release];

    pipeline = nullptr;
    paramsBuffer = nullptr;
  }

  void MetalInputProcess::finalize()
  {
    NSError* error = nil;

    id<MTLDevice> device = engine->getMTLDevice();

    pipeline = engine->newMTLComputePipelineState("input_process");

    paramsBuffer = [device newBufferWithLength: sizeof(ProcessParams)
                                       options: MTLResourceStorageModeShared];
  }

  void MetalInputProcess::submit()
  {
    if (!getMainSrc() || !dst)
      throw std::logic_error("input processing source/destination not set");
    if (tile.hSrcBegin + tile.H > getMainSrc()->getH() ||
        tile.wSrcBegin + tile.W > getMainSrc()->getW() ||
        tile.hDstBegin + tile.H > dst->getH() ||
        tile.wDstBegin + tile.W > dst->getW())
      throw std::out_of_range("input processing source/destination out of range");

    ProcessParams params = createProcessParams();

    void* paramsPtr = [paramsBuffer contents];
    memcpy(paramsPtr, &params, sizeof(params));

    id<MTLBuffer> bufferColor = bufferFromImageOrMain(color.get());
    id<MTLBuffer> bufferAlbedo = bufferFromImageOrMain(albedo.get());
    id<MTLBuffer> bufferNormal = bufferFromImageOrMain(normal.get());
    id<MTLBuffer> bufferOutput = getMTLBuffer(dst->getBuffer());

    auto commandBuffer = engine->getMTLCommandBuffer();
    auto computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState: pipeline];

    int index = 0;

    [computeEncoder setBuffer: bufferColor
                       offset: 0
                      atIndex: index++];

    [computeEncoder setBuffer: bufferAlbedo
                       offset: 0
                      atIndex: index++];

    [computeEncoder setBuffer: bufferNormal
                       offset: 0
                      atIndex: index++];

    [computeEncoder setBuffer: bufferOutput
                       offset: 0
                      atIndex: index++];

    [computeEncoder setBuffer: paramsBuffer
                       offset: 0
                      atIndex: index++];

    auto width = pipeline.threadExecutionWidth;
    auto height = pipeline.maxTotalThreadsPerThreadgroup / width;
    auto threadsPerThreadgroup = MTLSizeMake(width, height, 1);
    auto threadgroupsPerGrid = MTLSizeMake((dstDesc.getW() + width - 1) / width,
                                           (dstDesc.getH() + height - 1) / height, 1);

    [computeEncoder dispatchThreadgroups: threadgroupsPerGrid
                   threadsPerThreadgroup: threadsPerThreadgroup];

    [computeEncoder endEncoding];
    [commandBuffer commit];
  }

  ProcessParams MetalInputProcess::createProcessParams()
  {
    auto src = getMainSrc();
    ProcessParams params = {0};
    params.C = dst->getC();
    params.H = src->getH();
    params.W = src->getW();
    params.tile.H = tile.H;
    params.tile.W = tile.W;
    params.tile.hSrcBegin = tile.hSrcBegin;
    params.tile.wSrcBegin = tile.wSrcBegin;
    params.tile.hDstBegin = tile.hDstBegin;
    params.tile.wDstBegin = tile.wDstBegin;
    params.hdr = hdr;
    params.snorm = snorm;
    params.func = toTransferFunctionType(transferFunc->getType());
    params.inputScalePtr = transferFunc->inputScalePtr;
    params.inputScale = transferFunc->inputScale;
    params.outputScale = transferFunc->outputScale;
    params.normScale = transferFunc->getNormScale();
    params.color = color ? true : false;
    params.albedo = albedo ? true : false;
    params.normal = normal ? true : false;
    params.inputDataType = toDataType(getMainSrc()->getDataType());
    params.outputDataType = toDataType(dst->getDataType());
    return params;
  }

  id<MTLBuffer> MetalInputProcess::bufferFromImageOrMain(Image* image)
  {
    if (image)
      return getMTLBuffer(image->getBuffer());
    else
      return getMTLBuffer(getMainSrc()->getBuffer());
  }

OIDN_NAMESPACE_END
