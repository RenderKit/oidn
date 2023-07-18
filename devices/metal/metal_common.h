// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/tensor.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

OIDN_NAMESPACE_BEGIN

  class MetalDevice;
  struct PoolDesc;
  struct ConvDesc;

  MPSDataType toMPSDataType(DataType dataType);
  MPSShape* toMPSShape(const TensorDesc& td);

  MPSGraphTensor* toMPSGraphTensor(MPSGraph* graph, const std::shared_ptr<Tensor>& t);
  MPSGraphTensor* toMPSGraphPlaceholder(MPSGraph* graph, TensorDesc td);
  MPSGraphTensor* toMPSGraphPlaceholder(MPSGraph* graph, ImageDesc imd);
  MPSGraphTensorData* newMPSGraphTensorData(const std::shared_ptr<Tensor>& tensor);

  id<MTLBuffer> getMTLBuffer(Ref<Buffer> buffer);

OIDN_NAMESPACE_END
