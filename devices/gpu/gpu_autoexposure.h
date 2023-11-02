// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/autoexposure.h"
#include "core/color.h"
#include "core/kernel.h"

OIDN_NAMESPACE_BEGIN

  template<int maxBinSize>
  struct GPUAutoexposureDownsampleKernel : WorkGroup<2>
  {
    ImageAccessor src;
    float* bins;

    OIDN_DEVICE_INLINE void operator ()(const WorkGroupItem<2>& it) const
    {
      constexpr int groupSize = maxBinSize * maxBinSize;
      OIDN_SHARED LocalArray<float, groupSize> localSums;

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
      localSums[localID] = L;

      for (int i = groupSize / 2; i > 0; i >>= 1)
      {
        it.groupBarrier();
        if (localID < i)
          localSums[localID] += localSums[localID + i];
      }

      if (localID == 0)
      {
        const float avgL = localSums[0] / float((endH - beginH) * (endW - beginW));
        bins[it.getGroupLinearID()] = avgL;
      }
    }
  };

  template<int groupSize>
  struct GPUAutoexposureReduceKernel : WorkGroup<1>
  {
    const float* bins;
    int size;
    float* sums;
    int* counts;

    OIDN_DEVICE_INLINE void operator ()(const WorkGroupItem<1>& it) const
    {
      OIDN_SHARED LocalArray<float, groupSize> localSums;
      OIDN_SHARED LocalArray<int, groupSize> localCounts;

      float sum = 0;
      int count = 0;
      for (int i = it.getGlobalID(); i < size; i += it.getGlobalSize())
      {
        const float L = bins[i];
        if (L > Autoexposure::eps)
        {
          sum += math::log2(L);
          ++count;
        }
      }

      const int localID = it.getLocalID();
      localSums[localID] = sum;
      localCounts[localID] = count;

      for (int i = groupSize / 2; i > 0; i >>= 1)
      {
        it.groupBarrier();
        if (localID < i)
        {
          localSums[localID] += localSums[localID + i];
          localCounts[localID] += localCounts[localID + i];
        }
      }

      if (localID == 0)
      {
        sums[it.getGroupID()] = localSums[0];
        counts[it.getGroupID()] = localCounts[0];
      }
    }
  };

  template<int groupSize>
  struct GPUAutoexposureReduceFinalKernel : WorkGroup<1>
  {
    const float* sums;
    const int* counts;
    int size;
    float* result;

    OIDN_DEVICE_INLINE void operator ()(const WorkGroupItem<1>& it) const
    {
      OIDN_SHARED LocalArray<float, groupSize> localSums;
      OIDN_SHARED LocalArray<int, groupSize> localCounts;

      const int localID = it.getLocalID();

      if (localID < size)
      {
        localSums[localID] = sums[localID];
        localCounts[localID] = counts[localID];
      }
      else
      {
        localSums[localID] = 0;
        localCounts[localID] = 0;
      }

      for (int i = groupSize / 2; i > 0; i >>= 1)
      {
        it.groupBarrier();
        if (localID < i)
        {
          localSums[localID] += localSums[localID + i];
          localCounts[localID] += localCounts[localID + i];
        }
      }

      if (localID == 0)
        *result = (localCounts[0] > 0) ? (Autoexposure::key / math::exp2(localSums[0] / float(localCounts[0]))) : 1.f;
    }
  };

  template<typename EngineT, int groupSize>
  class GPUAutoexposure final : public Autoexposure
  {
    static_assert(groupSize >= maxBinSize * maxBinSize, "GPUAutoexposure groupSize is too small");

  public:
    GPUAutoexposure(const Ref<EngineT>& engine, const ImageDesc& srcDesc)
      : Autoexposure(srcDesc),
        engine(engine)
    {
      numGroups = min(ceil_div(numBins, groupSize), groupSize);
      scratchByteSize = numBins * sizeof(float) + numGroups * (sizeof(float) + sizeof(int));
      resultBuffer = engine->newBuffer(sizeof(float), Storage::Device);
    }

    size_t getScratchByteSize() const override
    {
      return scratchByteSize;
    }

    void setScratch(const Ref<Buffer>& scratch) override
    {
      if (scratch->getByteSize() < getScratchByteSize())
        throw std::invalid_argument("autoexposure scratch buffer too small");
      this->scratch = scratch;
    }

    void submit() override
    {
      if (!src)
        throw std::logic_error("autoexposure source not set");
      if (!scratch)
        throw std::logic_error("autoexposure scratch not set");

      float* bins = (float*)scratch->getData();
      float* sums = (float*)((char*)bins + numBins * sizeof(float));
      int* counts = (int*)((char*)sums + numGroups * sizeof(float));

      GPUAutoexposureDownsampleKernel<maxBinSize> downsample;
      downsample.src = *src;
      downsample.bins = bins;
      engine->submitKernel(WorkDim<2>(numBinsH, numBinsW), WorkDim<2>(maxBinSize, maxBinSize), downsample);

      GPUAutoexposureReduceKernel<groupSize> reduce;
      reduce.bins   = bins;
      reduce.size   = numBins;
      reduce.sums   = sums;
      reduce.counts = counts;
      engine->submitKernel(WorkDim<1>(numGroups), WorkDim<1>(groupSize), reduce);

      GPUAutoexposureReduceFinalKernel<groupSize> reduceFinal;
      reduceFinal.sums   = sums;
      reduceFinal.counts = counts;
      reduceFinal.size   = numGroups;
      reduceFinal.result = (float*)resultBuffer->getData();
      engine->submitKernel(WorkDim<1>(1), WorkDim<1>(groupSize), reduceFinal);
    }

    const float* getResult() const override { return (float*)resultBuffer->getData(); }

  private:
    Ref<EngineT> engine;
    int numGroups;
    Ref<Buffer> resultBuffer;
    size_t scratchByteSize;
    Ref<Buffer> scratch;
  };

OIDN_NAMESPACE_END
