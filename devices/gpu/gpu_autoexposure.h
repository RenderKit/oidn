// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/kernel.h"
#include "core/image_accessor.h"
#include "core/color.h"
#include "core/autoexposure.h"

OIDN_NAMESPACE_BEGIN

  template<int maxBinSize>
  struct GPUAutoexposureDownsampleKernel
  {
    static constexpr oidn_constant int groupSize = maxBinSize * maxBinSize;

    ImageAccessor src;
    oidn_global float* bins;

    // Shared local memory
    struct Local
    {
      float sums[groupSize];
    };

    oidn_device_inline void operator ()(const oidn_private WorkGroupItem<2>& it, LocalPtr<Local> local) const
    {
      const int beginH = it.getGroupID<0>() * src.H / it.getNumGroups<0>();
      const int beginW = it.getGroupID<1>() * src.W / it.getNumGroups<1>();
      const int endH = (it.getGroupID<0>()+1) * src.H / it.getNumGroups<0>();
      const int endW = (it.getGroupID<1>()+1) * src.W / it.getNumGroups<1>();

      const int h = beginH + it.getLocalID<0>();
      const int w = beginW + it.getLocalID<1>();

      float L;
      if (h < endH && w < endW)
      {
        vec3f c = src.get3(h, w);
        c = math::clamp(math::nan_to_zero(c), 0.f, FLT_MAX); // sanitize
        L = luminance(c);
      }
      else
      {
        L = 0;
      }

      const int localID = it.getLocalLinearID();
      local->sums[localID] = L;

      for (int i = groupSize / 2; i > 0; i >>= 1)
      {
        it.groupBarrier();
        if (localID < i)
          local->sums[localID] += local->sums[localID + i];
      }

      if (localID == 0)
      {
        const float avgL = local->sums[0] / float((endH - beginH) * (endW - beginW));
        bins[it.getGroupLinearID()] = avgL;
      }
    }
  };

  template<int groupSize>
  struct GPUAutoexposureReduceKernel
  {
    const oidn_global float* bins;
    int size;
    oidn_global float* sums;
    oidn_global int* counts;

    // Shared local memory
    struct Local
    {
      float sums[groupSize];
      int counts[groupSize];
    };

    oidn_device_inline void operator ()(const oidn_private WorkGroupItem<1>& it, LocalPtr<Local> local) const
    {
      float sum = 0;
      int count = 0;
      for (int i = it.getGlobalID(); i < size; i += it.getGlobalSize())
      {
        const float L = bins[i];
        if (L > AutoexposureParams::eps)
        {
          sum += math::log2(L);
          ++count;
        }
      }

      const int localID = it.getLocalID();
      local->sums[localID]   = sum;
      local->counts[localID] = count;

      for (int i = groupSize / 2; i > 0; i >>= 1)
      {
        it.groupBarrier();
        if (localID < i)
        {
          local->sums[localID]   += local->sums[localID + i];
          local->counts[localID] += local->counts[localID + i];
        }
      }

      if (localID == 0)
      {
        sums[it.getGroupID()]   = local->sums[0];
        counts[it.getGroupID()] = local->counts[0];
      }
    }
  };

  template<int groupSize>
  struct GPUAutoexposureReduceFinalKernel
  {
    const oidn_global float* sums;
    const oidn_global int* counts;
    int size;
    oidn_global float* result;

    // Shared local memory
    struct Local
    {
      float sums[groupSize];
      int counts[groupSize];
    };

    oidn_device_inline void operator ()(const oidn_private WorkGroupItem<1>& it, LocalPtr<Local> local) const
    {
      const int localID = it.getLocalID();

      if (localID < size)
      {
        local->sums[localID]   = sums[localID];
        local->counts[localID] = counts[localID];
      }
      else
      {
        local->sums[localID]   = 0;
        local->counts[localID] = 0;
      }

      for (int i = groupSize / 2; i > 0; i >>= 1)
      {
        it.groupBarrier();
        if (localID < i)
        {
          local->sums[localID]   += local->sums[localID + i];
          local->counts[localID] += local->counts[localID + i];
        }
      }

      if (localID == 0)
      {
        *result = (local->counts[0] > 0) ?
                  (AutoexposureParams::key / math::exp2(local->sums[0] / float(local->counts[0]))) : 1.f;
      }
    }
  };

#if !defined(OIDN_COMPILE_METAL_DEVICE)

  template<typename EngineT, int groupSize>
  class GPUAutoexposure final : public Autoexposure
  {
    static_assert(groupSize >= maxBinSize * maxBinSize, "GPUAutoexposure groupSize is too small");

  public:
    GPUAutoexposure(EngineT* engine, const ImageDesc& srcDesc)
      : Autoexposure(srcDesc),
        engine(engine)
    {
      numGroups = min(ceil_div(numBins, groupSize), groupSize);
      scratchByteSize = numBins * sizeof(float) + numGroups * (sizeof(float) + sizeof(int));
    }

    Engine* getEngine() const override { return engine; }

  #if defined(OIDN_COMPILE_METAL)
    void finalize() override
    {
      downsamplePipeline = engine->newPipeline("autoexposureDownsample");
      reducePipeline = engine->newPipeline("autoexposureReduce_" + toString(groupSize));
      reduceFinalPipeline = engine->newPipeline("autoexposureReduceFinal_" + toString(groupSize));
    }
  #endif

    size_t getScratchByteSize() override
    {
      return scratchByteSize;
    }

    void setScratch(const Ref<Buffer>& scratch) override
    {
      if (scratch->getByteSize() < getScratchByteSize())
        throw std::invalid_argument("autoexposure scratch buffer too small");
      this->scratch = scratch;
    }

    void submitKernels(const Ref<CancellationToken>& ct) override
    {
      if (!src)
        throw std::logic_error("autoexposure source not set");
      if (!dst)
        throw std::logic_error("autoexposure destination not set");
      if (!scratch)
        throw std::logic_error("autoexposure scratch not set");
      if (dst->getBuffer() != scratch)
        throw std::invalid_argument("autoexposure result must be stored in the scratch buffer");

      float* bins = (float*)scratch->getPtr();
      float* sums = (float*)((char*)bins + numBins * sizeof(float));
      int* counts = (int*)((char*)sums + numGroups * sizeof(float));

      GPUAutoexposureDownsampleKernel<maxBinSize> downsample;
      downsample.src  = *src;
      downsample.bins = bins;

      GPUAutoexposureReduceKernel<groupSize> reduce;
      reduce.bins   = bins;
      reduce.size   = numBins;
      reduce.sums   = sums;
      reduce.counts = counts;

      GPUAutoexposureReduceFinalKernel<groupSize> reduceFinal;
      reduceFinal.sums   = sums;
      reduceFinal.counts = counts;
      reduceFinal.size   = numGroups;
      reduceFinal.result = getDstPtr();

    #if defined(OIDN_COMPILE_METAL)
      engine->submitKernel(WorkDim<2>(numBinsH, numBinsW), WorkDim<2>(maxBinSize, maxBinSize), downsample,
                           downsamplePipeline, {scratch, src->getBuffer()});

      engine->submitKernel(WorkDim<1>(numGroups), WorkDim<1>(groupSize), reduce,
                           reducePipeline, {scratch});

      engine->submitKernel(WorkDim<1>(1), WorkDim<1>(groupSize), reduceFinal,
                           reduceFinalPipeline, {scratch});
    #else
      engine->submitKernel(WorkDim<2>(numBinsH, numBinsW), WorkDim<2>(maxBinSize, maxBinSize), downsample);
      engine->submitKernel(WorkDim<1>(numGroups), WorkDim<1>(groupSize), reduce);
      engine->submitKernel(WorkDim<1>(1), WorkDim<1>(groupSize), reduceFinal);
    #endif
    }

  private:
    EngineT* engine;
    int numGroups;
    size_t scratchByteSize;
    Ref<Buffer> scratch;

  #if defined(OIDN_COMPILE_METAL)
    Ref<MetalPipeline> downsamplePipeline;
    Ref<MetalPipeline> reducePipeline;
    Ref<MetalPipeline> reduceFinalPipeline;
  #endif
  };

#endif

OIDN_NAMESPACE_END
