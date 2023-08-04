// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/engine.h"
#include "metal_device.h"

OIDN_NAMESPACE_BEGIN

  class MetalConcat;

  class MetalEngine : public Engine
  {
  public:
    explicit MetalEngine(const Ref<MetalDevice>& device);
    ~MetalEngine();

    Device* getDevice() const override { return device; }

    // Metal
    id<MTLDevice> getMTLDevice() const { return device->getMTLDevice(); }
    id<MTLCommandQueue> getMTLCommandQueue() const { return commandQueue; }
    id<MTLComputePipelineState> newMTLComputePipelineState(const std::string& kernelName);
    id<MTLCommandBuffer> getMTLCommandBuffer();
    MPSCommandBuffer* getMPSCommandBuffer();

    // Buffer
    Ref<Buffer> newBuffer(size_t byteSize, Storage storage) override;
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize) override;

    // Ops
    std::shared_ptr<Graph> newGraph(const std::shared_ptr<TensorMap>& constTensors, bool fastMath = false) override;
    std::shared_ptr<Conv> newConv(const ConvDesc& desc) override;
    std::shared_ptr<Pool> newPool(const PoolDesc& desc) override;
    std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) override;
    std::shared_ptr<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;
    std::shared_ptr<ImageCopy> newImageCopy() override;

    // Runs a parallel host task in the thread arena (if it exists)
    void runHostTask(std::function<void()>&& f) override;

    // Enqueues a basic kernel
    template<int N, typename Kernel>
    OIDN_INLINE void submitKernel(WorkDim<N> globalSize, const Kernel& kernel,
                                  id<MTLComputePipelineState> pipeline,
                                  const std::vector<id<MTLBuffer>>& buffers)
    {
      auto commandBuffer = getMTLCommandBuffer();
      auto computeEncoder = [commandBuffer computeCommandEncoder];

      [computeEncoder setComputePipelineState: pipeline];

      [computeEncoder setBytes: &kernel
                        length: sizeof(kernel)
                       atIndex: 0];

      for (auto buffer : buffers)
      {
        if (buffer)
        {
          [computeEncoder useResource: buffer
                                usage: MTLResourceUsageRead | MTLResourceUsageWrite];
        }
      }

      // Metal 3 devices support non-uniform threadgroup sizes
      // FIXME: improve threadsPerThreadgroup
      [computeEncoder dispatchThreads: MTLSize(globalSize)
                threadsPerThreadgroup: MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];

      [computeEncoder endEncoding];
      [commandBuffer commit];
    }

    // Enqueues a work-group kernel
    template<int N, typename Kernel>
    OIDN_INLINE void submitKernel(WorkDim<N> numGroups, WorkDim<N> groupSize, const Kernel& kernel,
                                  id<MTLComputePipelineState> pipeline,
                                  const std::vector<id<MTLBuffer>>& buffers)
    {
      auto commandBuffer = getMTLCommandBuffer();
      auto computeEncoder = [commandBuffer computeCommandEncoder];

      [computeEncoder setComputePipelineState: pipeline];

      [computeEncoder setBytes: &kernel
                        length: sizeof(kernel)
                       atIndex: 0];

      for (auto buffer : buffers)
      {
        if (buffer)
        {
          [computeEncoder useResource: buffer
                                usage: MTLResourceUsageRead | MTLResourceUsageWrite];
        }
      }

      [computeEncoder dispatchThreadgroups: MTLSize(numGroups)
                     threadsPerThreadgroup: MTLSize(groupSize)];

      [computeEncoder endEncoding];
      [commandBuffer commit];
    }

    // Enqueues a host function
    void submitHostFunc(std::function<void()>&& f) override;

    void wait() override;

  private:
    MetalDevice* device;
    id<MTLCommandQueue> commandQueue;
    id<MTLCommandBuffer> lastCommandBuffer;
    id<MTLLibrary> library;
  };

OIDN_NAMESPACE_END
