// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/tensor.h"
#include "core/color.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_kernel_common.h"

OIDN_NAMESPACE_BEGIN

  class MetalDevice;
  struct PoolDesc;
  struct ConvDesc;

  id<MTLDevice> mtlDevice(int deviceID);

  MPSDataType toMPSDataType(DataType dataType);
  MPSShape* toMPSShape(const TensorDesc& td);

  MPSGraphTensor* toMPSGraphTensor(MPSGraph* graph, const std::shared_ptr<Tensor>& t);
  MPSGraphTensor* toMPSGraphPlaceholder(MPSGraph* graph, TensorDesc td);
  MPSGraphTensor* toMPSGraphPlaceholder(MPSGraph* graph, ImageDesc imd);
  MPSGraphTensorData* newMPSGraphTensorData(const std::shared_ptr<Tensor>& tensor);

  TransferFunctionType toTransferFunctionType(TransferFunction::Type type);
  KernelDataType toDataType(DataType type);

  id<MTLBuffer> getMTLBuffer(Ref<Buffer> buffer);

OIDN_NAMESPACE_END
