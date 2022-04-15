// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_autoexposure.h"
#include "../color.h"

namespace oidn {

  CPUAutoexposure::CPUAutoexposure(const Ref<Device>& device, const ImageDesc& srcDesc)
    : BaseOp(device),
      Autoexposure(srcDesc),
      result(0) {}

  void CPUAutoexposure::run()
  {
    constexpr float key = 0.18f;
    constexpr float eps = 1e-8f;

    // Downsample the image to minimize sensitivity to noise
    ispc::ImageAccessor srcAcc = *src;

    // Compute the average log luminance of the downsampled image
    using Sum = std::pair<float, int>;

    Sum sum =
      tbb::parallel_reduce(
        tbb::blocked_range2d<int>(0, numBinsH, 0, numBinsW),
        Sum(0.f, 0),
        [&](const tbb::blocked_range2d<int>& r, Sum sum) -> Sum
        {
          // Iterate over bins
          for (int i = r.rows().begin(); i != r.rows().end(); ++i)
          {
            for (int j = r.cols().begin(); j != r.cols().end(); ++j)
            {
              // Compute the average luminance in the current bin
              const int beginH = int(ptrdiff_t(i)   * src->getH() / numBinsH);
              const int beginW = int(ptrdiff_t(j)   * src->getW() / numBinsW);
              const int endH   = int(ptrdiff_t(i+1) * src->getH() / numBinsH);
              const int endW   = int(ptrdiff_t(j+1) * src->getW() / numBinsW);

              const float L = ispc::getAvgLuminance(srcAcc, beginH, endH, beginW, endW);

              // Accumulate the log luminance
              if (L > eps)
              {
                sum.first += log2(L);
                sum.second++;
              }
            }
          }

          return sum;
        },
        [](Sum a, Sum b) -> Sum { return Sum(a.first+b.first, a.second+b.second); }
      );

    result = (sum.second > 0) ? (key / exp2(sum.first / float(sum.second))) : 1.f;
  }

}