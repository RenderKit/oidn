// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_common.h"
#include "metal_engine.h"
#include "metal_buffer.h"
#include "core/scratch.h"

OIDN_NAMESPACE_BEGIN

  id<MTLDevice> mtlDevice(int deviceID)
  {
    NSArray* devices = [MTLCopyAllDevices() autorelease];
    int numDevice = (int)[devices count];
    assert(deviceID < numDevice);
    return [devices[deviceID] retain];
  }

  MPSDataType_t toMPSDataType(DataType dataType) {
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

  MPSShape_t toMPSShape(const TensorDesc& td)
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

  MPSGraphTensor_t toMPSGraphTensor(MPSGraph_t graph, const std::shared_ptr<Tensor>& t)
  {
    NSData* data = [NSData dataWithBytes: t->getPtr()
                                  length: t->getByteSize()];

    return [graph constantWithData: data
                             shape: toMPSShape(t->getDesc())
                          dataType: toMPSDataType(t->getDataType())];
  }

  MPSGraphTensor_t toMPSGraphPlaceholder(MPSGraph_t graph, TensorDesc td)
  {
    return [graph placeholderWithShape: toMPSShape(td)
                              dataType: toMPSDataType(td.dataType)
                                  name: nil];
  }

  MPSGraphTensor_t toMPSGraphPlaceholder(MPSGraph_t graph, ImageDesc imd)
  {
    return [graph placeholderWithShape: @[@(imd.getH()), @(imd.getW()), @(imd.getC())]
                              dataType: toMPSDataType(imd.getDataType())
                                  name: nil];
  }

  MPSGraphTensorData_t toMPSGraphTensorData(id<MTLBuffer> buffer, const std::shared_ptr<Tensor>& t)
  {
    return toMPSGraphTensorData(buffer, t->getDesc());
  }

  MPSGraphTensorData_t toMPSGraphTensorData(id<MTLBuffer> buffer, TensorDesc td)
  {
    return [[MPSGraphTensorData alloc] initWithMTLBuffer: buffer
                                                   shape: toMPSShape(td)
                                                dataType: toMPSDataType(td.dataType)];
  }

  MPSGraphTensorData_t toMPSGraphTensorData(id<MTLBuffer> buffer, ImageDesc imd)
  {
    return [[MPSGraphTensorData alloc] initWithMTLBuffer: buffer
                                                   shape: @[@1, @(imd.getH()), @(imd.getW()), @(imd.getC())]
                                                dataType: toMPSDataType(imd.getDataType())];
  }

  TransferFunctionType toTransferFunctionType(TransferFunction::Type type)
  {
    switch (type) {
      case TransferFunction::Type::PU:
        return TransferFunctionType::PU;
      case TransferFunction::Type::Log:
        return TransferFunctionType::Log;
      case TransferFunction::Type::Linear:
        return TransferFunctionType::Linear;
      case TransferFunction::Type::SRGB:
        return TransferFunctionType::SRGB;
      default:
        throw std::logic_error("unknown transfer function");
    }
  }

  KernelDataType toDataType(DataType type)
  {
    switch (type) {
      case DataType::Float32:
        return KernelDataType::f32;
      case DataType::Float16:
        return KernelDataType::f16;
      default:
        throw std::logic_error("unknown data type");
    }
  }

  id<MTLBuffer> getMTLBuffer(Ref<Buffer> buffer)
  {
    if (!buffer)
      return nil;

    Buffer* baseBuffer = buffer.get();
    if (ScratchBuffer* scratchBuffer = dynamic_cast<ScratchBuffer*>(baseBuffer))
      baseBuffer = scratchBuffer->getParentBuffer();

    if (MetalBuffer* metalBuffer = dynamic_cast<MetalBuffer*>(baseBuffer))
      return metalBuffer->getMTLBuffer();
    else
      throw std::logic_error("buffer is not a Metal buffer");
  }

OIDN_NAMESPACE_END
