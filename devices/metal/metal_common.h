// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/tensor.h"
#include "core/color.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
typedef id<MTLLibrary> MTLLibrary_t;
typedef id<MTLComputePipelineState> MTLComputePipelineState_t;
typedef id<MTLLibrary> MTLLibrary_t;
typedef id<MTLCommandQueue> MTLCommandQueue_t;
typedef MPSGraph* MPSGraph_t;
typedef MPSGraphTensor* MPSGraphTensor_t;
typedef MPSGraphTensorData* MPSGraphTensorData_t;
typedef MPSGraphPooling2DOpDescriptor* MPSGraphPooling2DOpDescriptor_t;
typedef MPSGraphConvolution2DOpDescriptor* MPSGraphConvolution2DOpDescriptor_t;
typedef MPSShape* MPSShape_t;
typedef NSData* NSData_t;
typedef MPSDataType MPSDataType_t;

#include "metal_kernel_common.h"

OIDN_NAMESPACE_BEGIN

  class MetalDevice;
  struct PoolDesc;
  struct ConvDesc;

  id<MTLDevice> mtlDevice(int deviceID);

  MPSDataType_t toMPSDataType(DataType dataType);
  MPSShape_t toMPSShape(const TensorDesc& td);

  MPSGraphTensor_t toMPSGraphTensor(MPSGraph_t graph, const std::shared_ptr<Tensor>& t);
  MPSGraphTensor_t toMPSGraphPlaceholder(MPSGraph_t graph, TensorDesc td);
  MPSGraphTensor_t toMPSGraphPlaceholder(MPSGraph_t graph, ImageDesc imd);
  MPSGraphTensorData_t toMPSGraphTensorData(id<MTLBuffer> buffer, const std::shared_ptr<Tensor>& t);
  MPSGraphTensorData_t toMPSGraphTensorData(id<MTLBuffer> buffer, TensorDesc td);
  MPSGraphTensorData_t toMPSGraphTensorData(id<MTLBuffer> buffer, ImageDesc imd);

  MPSGraphPooling2DOpDescriptor_t MPSGraphPoolDesc();
  MPSGraphConvolution2DOpDescriptor_t MPSGraphConvDesc();

  TransferFunctionType toTransferFunctionType(TransferFunction::Type type);
  KernelDataType toDataType(DataType type);

  id<MTLBuffer> getMTLBuffer(Ref<Buffer> buffer);

  MTLComputePipelineState_t createPipeline(id<MTLDevice> device, std::string function);

OIDN_NAMESPACE_END
