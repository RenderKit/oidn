// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../autoexposure.h"
#include "../color.h"
#include "../kernel.h"

namespace oidn {

  template<typename ImageT, int maxBinSize>
  struct GPUAutoexposureDownsampleKernel : WorkGroup<2>
  {
    ImageAccessor<ImageT> src;
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
        c = clamp(nan_to_zero(c), 0.f, FLT_MAX); // sanitize
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
      constexpr float eps = 1e-8f;

      OIDN_SHARED LocalArray<float, groupSize> localSums;
      OIDN_SHARED LocalArray<int, groupSize> localCounts;

      float sum = 0;
      int count = 0;
      for (int i = it.getGlobalId(); i < size; i += it.getGlobalRange())
      {
        const float L = bins[i];
        if (L > eps)
        {
          sum += log2(L);
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
      {
        constexpr float key = 0.18f;
        *result = (localCounts[0] > 0) ? (key / exp2(localSums[0] / float(localCounts[0]))) : 1.f;
      }
    }
  };

  template<typename OpT>
  class GPUAutoexposure final : public OpT, public Autoexposure
  {
  public:
    GPUAutoexposure(const Ref<typename OpT::DeviceType>& device, const ImageDesc& srcDesc)
      : OpT(device),
        Autoexposure(srcDesc)
    {
      numGroups = min(ceil_div(numBins, groupSize), groupSize);
      scratchSize = numBins * sizeof(float) + numGroups * (sizeof(float) + sizeof(int));
      resultBuffer = device->newBuffer(sizeof(float), Storage::Device);
    }

    size_t getScratchByteSize() const override
    {
      return scratchSize;
    }

    void setScratch(const std::shared_ptr<Tensor>& scratch) override
    {
      assert(scratch->getByteSize() >= scratchSize);
      this->scratch = scratch;
    }

    void run() override
    {
      switch (src->getDataType())
      {
      case DataType::Float32:
        runKernel<float>();
        break;
      case DataType::Float16:
        runKernel<half>();
        break;
      default:
        assert(0);
      }
    }

    const float* getResult() const override { return (float*)resultBuffer->getData(); }

  private:
    template<typename T>
    void runKernel()
    {
      float* bins = (float*)scratch->getData();
      float* sums = (float*)((char*)bins + numBins * sizeof(float));
      int* counts = (int*)((char*)sums + numGroups * sizeof(float));

      GPUAutoexposureDownsampleKernel<T, maxBinSize> downsample;
      downsample.src = *src;
      downsample.bins = bins;
      this->device->runKernel({numBinsH, numBinsW}, {maxBinSize, maxBinSize}, downsample);

      GPUAutoexposureReduceKernel<groupSize> reduce;
      reduce.bins   = bins;
      reduce.size   = numBins;
      reduce.sums   = sums;
      reduce.counts = counts;
      this->device->runKernel(numGroups, groupSize, reduce);

      GPUAutoexposureReduceFinalKernel<groupSize> reduceFinal;
      reduceFinal.sums   = sums;
      reduceFinal.counts = counts;
      reduceFinal.size   = numGroups;
      reduceFinal.result = (float*)resultBuffer->getData();
      this->device->runKernel(1, groupSize, reduceFinal);
    }

    static constexpr int groupSize = 1024;
    int numGroups;
    Ref<Buffer> resultBuffer;
    size_t scratchSize;
    std::shared_ptr<Tensor> scratch;
  };

} // namespace oidn
