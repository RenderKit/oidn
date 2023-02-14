// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/autoexposure.h"
#include "core/color.h"
#include "core/kernel.h"

OIDN_NAMESPACE_BEGIN

  template<typename ImageDataT, int maxBinSize>
  struct GPUAutoexposureDownsampleKernel : WorkGroup<2>
  {
    ImageAccessor<ImageDataT> src;
    float* bins;

    OIDN_DEVICE_INLINE void operator ()(const WorkGroupItem<2>& it) const
    {
      constexpr int groupSize = maxBinSize * maxBinSize;
      OIDN_SHARED LocalArray<float, groupSize> localSums;

      const int beginH = it.getGroupId<0>() * src.H / it.getGroupRange<0>();
      const int beginW = it.getGroupId<1>() * src.W / it.getGroupRange<1>();
      const int endH = (it.getGroupId<0>()+1) * src.H / it.getGroupRange<0>();
      const int endW = (it.getGroupId<1>()+1) * src.W / it.getGroupRange<1>();

      const int h = beginH + it.getLocalId<0>();
      const int w = beginW + it.getLocalId<1>();

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

      const int localId = it.getLocalLinearId();
      localSums[localId] = L;

      for (int i = groupSize / 2; i > 0; i >>= 1)
      {
        it.syncGroup();
        if (localId < i)
          localSums[localId] += localSums[localId + i];
      }

      if (localId == 0)
      {
        const float avgL = localSums[0] / float((endH - beginH) * (endW - beginW));
        bins[it.getGroupLinearId()] = avgL;
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
      for (int i = it.getGlobalId(); i < size; i += it.getGlobalRange())
      {
        const float L = bins[i];
        if (L > Autoexposure::eps)
        {
          sum += math::log2(L);
          ++count;
        }
      }

      const int localId = it.getLocalId();
      localSums[localId] = sum;
      localCounts[localId] = count;

      for (int i = groupSize / 2; i > 0; i >>= 1)
      {
        it.syncGroup();
        if (localId < i)
        {
          localSums[localId] += localSums[localId + i];
          localCounts[localId] += localCounts[localId + i];
        }
      }

      if (localId == 0)
      {
        sums[it.getGroupId()] = localSums[0];
        counts[it.getGroupId()] = localCounts[0];
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

      const int localId = it.getLocalId();

      if (localId < size)
      {
        localSums[localId] = sums[localId];
        localCounts[localId] = counts[localId];
      }
      else
      {
        localSums[localId] = 0;
        localCounts[localId] = 0;
      }

      for (int i = groupSize / 2; i > 0; i >>= 1)
      {
        it.syncGroup();
        if (localId < i)
        {
          localSums[localId] += localSums[localId + i];
          localCounts[localId] += localCounts[localId + i];
        }
      }

      if (localId == 0)
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

    void setScratch(const std::shared_ptr<Tensor>& scratch) override
    {
      if (!scratch || scratch->getByteSize() < scratchByteSize)
        throw std::invalid_argument("invalid autoexposure scratch");
        
      this->scratch = scratch;
    }

    void submit() override
    {
      if (!src)
        throw std::logic_error("autoexposure source not set");
      if (!scratch)
        throw std::logic_error("autoexposure scratch not set");

      switch (src->getDataType())
      {
      case DataType::Float32: runImpl<float>(); break;
      case DataType::Float16: runImpl<half>();  break;
      default:                assert(0);
      }
    }

    const float* getResult() const override { return (float*)resultBuffer->getData(); }

  private:
    template<typename ImageDataT>
    void runImpl()
    {
      float* bins = (float*)scratch->getData();
      float* sums = (float*)((char*)bins + numBins * sizeof(float));
      int* counts = (int*)((char*)sums + numGroups * sizeof(float));

      GPUAutoexposureDownsampleKernel<ImageDataT, maxBinSize> downsample;
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

    Ref<EngineT> engine;
    int numGroups;
    Ref<Buffer> resultBuffer;
    size_t scratchByteSize;
    std::shared_ptr<Tensor> scratch;
  };

OIDN_NAMESPACE_END
