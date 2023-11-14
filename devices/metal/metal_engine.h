// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/engine.h"
#include "metal_device.h"

OIDN_NAMESPACE_BEGIN

  struct MetalPipeline : public RefCount
  {
  public:
    explicit MetalPipeline(id<MTLComputePipelineState> state) : state(state) {}
    ~MetalPipeline() { [state release]; }

    id<MTLComputePipelineState> getMTLPipelineState() const { return state; }
    int getSubgroupSize() const { return state.threadExecutionWidth; }

  private:
    id<MTLComputePipelineState> state;
  };

  class MetalEngine : public Engine
  {
  public:
    explicit MetalEngine(const Ref<MetalDevice>& device);
    ~MetalEngine();

    Device* getDevice() const override { return device; }

    // Metal
    id<MTLDevice> getMTLDevice() const { return device->getMTLDevice(); }
    id<MTLCommandBuffer> getMTLCommandBuffer();
    MPSCommandBuffer* getMPSCommandBuffer();

    // Heap
    Ref<Heap> newHeap(size_t byteSize, Storage storage) override;

    // Buffer
    SizeAndAlignment getBufferByteSizeAndAlignment(size_t byteSize, Storage storage) override;
    Ref<Buffer> newBuffer(size_t byteSize, Storage storage) override;
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize) override;
    Ref<Buffer> newBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset) override;

    // Tensor
    Ref<Tensor> newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset) override;

    // Ops
    bool isConvSupported(PostOp postOp) override;
    std::shared_ptr<Conv> newConv(const ConvDesc& desc) override;
    std::shared_ptr<Pool> newPool(const PoolDesc& desc) override;
    std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) override;
    std::shared_ptr<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;
    std::shared_ptr<ImageCopy> newImageCopy() override;

    // Runs a parallel host task in the thread arena (if it exists)
    void runHostTask(std::function<void()>&& f) override;

    // Creates a compute pipeline for executing a kernel with the given name
    Ref<MetalPipeline> newPipeline(const std::string& kernelName);

    // Enqueues a basic kernel
    template<int N, typename Kernel>
    OIDN_INLINE void submitKernel(WorkDim<N> globalSize, const Kernel& kernel,
                                  const Ref<MetalPipeline>& pipeline,
                                  const std::vector<Ref<Buffer>>& buffers)
    {
      auto commandBuffer = getMTLCommandBuffer();
      auto computeEncoder = [commandBuffer computeCommandEncoder];

      [computeEncoder setComputePipelineState: pipeline->getMTLPipelineState()];

      [computeEncoder setBytes: &kernel
                        length: sizeof(kernel)
                       atIndex: 0];

      for (auto buffer : buffers)
      {
        if (buffer)
        {
          [computeEncoder useResource: getMTLBuffer(buffer)
                                usage: MTLResourceUsageRead | MTLResourceUsageWrite];
        }
      }

      // Metal 3 devices support non-uniform threadgroup sizes
      // FIXME: improve threadsPerThreadgroup
      [computeEncoder dispatchThreads: MTLSize(globalSize)
                threadsPerThreadgroup: MTLSizeMake(pipeline->getSubgroupSize(), 1, 1)];

      [computeEncoder endEncoding];
    }

    // Enqueues a work-group kernel
    template<int N, typename Kernel>
    OIDN_INLINE void submitKernel(WorkDim<N> numGroups, WorkDim<N> groupSize, const Kernel& kernel,
                                  const Ref<MetalPipeline>& pipeline,
                                  const std::vector<Ref<Buffer>>& buffers)
    {
      auto commandBuffer = getMTLCommandBuffer();
      auto computeEncoder = [commandBuffer computeCommandEncoder];

      [computeEncoder setComputePipelineState: pipeline->getMTLPipelineState()];

      [computeEncoder setBytes: &kernel
                        length: sizeof(kernel)
                       atIndex: 0];

      for (auto buffer : buffers)
      {
        if (buffer)
        {
          [computeEncoder useResource: getMTLBuffer(buffer)
                                usage: MTLResourceUsageRead | MTLResourceUsageWrite];
        }
      }

      [computeEncoder dispatchThreadgroups: MTLSize(numGroups)
                     threadsPerThreadgroup: MTLSize(groupSize)];

      [computeEncoder endEncoding];
    }

    // Enqueues a host function
    void submitHostFunc(std::function<void()>&& f) override;

    void flush() override;
    void wait() override;

  private:
    MetalDevice* device;
    id<MTLCommandQueue> commandQueue;
    MPSCommandBuffer* commandBuffer;
    id<MTLLibrary> library;
  };

OIDN_NAMESPACE_END
