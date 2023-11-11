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

  MTLResourceOptions toMTLResourceOptions(Storage storage);

  MPSDataType toMPSDataType(DataType dataType);
  MPSShape* toMPSShape(const TensorDesc& td);

  MPSGraphTensor* toMPSGraphConst(MPSGraph* graph, const Ref<Tensor>& t);
  MPSGraphTensor* toMPSGraphPlaceholder(MPSGraph* graph, TensorDesc td);
  MPSGraphTensor* toMPSGraphPlaceholder(MPSGraph* graph, ImageDesc imd);
  MPSGraphTensorData* newMPSGraphTensorData(const Ref<Tensor>& tensor);

  id<MTLBuffer> getMTLBuffer(Ref<Buffer> buffer);

OIDN_NAMESPACE_END
