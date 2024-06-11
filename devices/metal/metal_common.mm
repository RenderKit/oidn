// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_common.h"
#include "metal_engine.h"
#include "metal_buffer.h"

OIDN_NAMESPACE_BEGIN

  MTLResourceOptions toMTLResourceOptions(Storage storage)
  {
    switch (storage)
    {
    case Storage::Host:
      return MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
    case Storage::Device:
      return MTLResourceStorageModePrivate;
  #if TARGET_OS_OSX || TARGET_OS_MACCATALYST
    case Storage::Managed:
      return MTLResourceStorageModeManaged | MTLResourceCPUCacheModeDefaultCache;
  #endif
    default:
      throw Exception(Error::InvalidArgument, "invalid storage mode");
    }
  }

  MPSDataType toMPSDataType(DataType dataType)
  {
    switch (dataType)
    {
    case DataType::Float32:
      return MPSDataType::MPSDataTypeFloat32;
    case DataType::Float16:
      return MPSDataType::MPSDataTypeFloat16;
    case DataType::UInt8:
      return MPSDataType::MPSDataTypeUInt8;
    default:
      throw std::invalid_argument("unsupported data type");
    }
  }

  MPSShape* toMPSShape(const TensorDesc& td)
  {
    switch (td.layout)
    {
    case TensorLayout::x:
      return @[@1, @1, @1, @(size_t(td.getX()))];
    case TensorLayout::hwc:
      return @[@1, @(size_t(td.getH())), @(size_t(td.getW())), @(size_t(td.getC()))];
    case TensorLayout::oihw:
      return @[@(size_t(td.getO())), @(size_t(td.getI())), @(size_t(td.getH())), @(size_t(td.getW()))];
    default:
      throw std::invalid_argument("unsupported tensor layout");
    }
  }

  MPSGraphTensor* toMPSGraphConst(MPSGraph* graph, const Ref<Tensor>& t)
  {
    NSData* data = [NSData dataWithBytes: t->getPtr()
                                  length: t->getByteSize()];

    return [graph constantWithData: data
                             shape: toMPSShape(t->getDesc())
                          dataType: toMPSDataType(t->getDataType())];
  }

  MPSGraphTensor* toMPSGraphPlaceholder(MPSGraph* graph, TensorDesc td)
  {
    return [graph placeholderWithShape: toMPSShape(td)
                              dataType: toMPSDataType(td.dataType)
                                  name: nil];
  }

  MPSGraphTensor* toMPSGraphPlaceholder(MPSGraph* graph, ImageDesc imd)
  {
    return [graph placeholderWithShape: @[@(imd.getH()), @(imd.getW()), @(imd.getC())]
                              dataType: toMPSDataType(imd.getDataType())
                                  name: nil];
  }

  MPSGraphTensorData* newMPSGraphTensorData(const Ref<Tensor>& tensor)
  {
    id<MTLBuffer> buffer = getMTLBuffer(tensor->getBuffer());
    if (tensor->getByteOffset() != 0)
      throw std::invalid_argument("MPSGraphTensorData requires zero offset in buffer");

    return [[MPSGraphTensorData alloc] initWithMTLBuffer: buffer
                                                   shape: toMPSShape(tensor->getDesc())
                                                dataType: toMPSDataType(tensor->getDataType())];
  }

  id<MTLBuffer> getMTLBuffer(Ref<Buffer> buffer)
  {
    if (!buffer)
      return nil;

    if (MetalBuffer* metalBuffer = dynamic_cast<MetalBuffer*>(buffer.get()))
      return metalBuffer->getMTLBuffer();
    else
      throw std::logic_error("buffer is not a Metal buffer");
  }

OIDN_NAMESPACE_END
