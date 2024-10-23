// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_autoexposure.h"
#include "cpu_autoexposure_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUAutoexposure::CPUAutoexposure(CPUEngine* engine, const ImageDesc& srcDesc)
    : Autoexposure(srcDesc),
      engine(engine)
  {}

  void CPUAutoexposure::submitKernels(const Ref<CancellationToken>& ct)
  {
    if (!src)
      throw std::logic_error("autoexposure source not set");
    if (!dst)
      throw std::logic_error("autoexposure destination not set");

    // Downsample the image to minimize sensitivity to noise
    ispc::ImageAccessor srcAcc = *src;
    float* dstPtr = getDstPtr();

    engine->submitFunc([=]()
    {
      // Compute the average log luminance of the downsampled image
      using Sum = std::pair<float, int>;

      Sum sum =
        tbb::parallel_deterministic_reduce(
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
                const int beginH = int(ptrdiff_t(i)   * srcAcc.H / numBinsH);
                const int beginW = int(ptrdiff_t(j)   * srcAcc.W / numBinsW);
                const int endH   = int(ptrdiff_t(i+1) * srcAcc.H / numBinsH);
                const int endW   = int(ptrdiff_t(j+1) * srcAcc.W / numBinsW);

                const float L = ispc::autoexposureDownsample(srcAcc, beginH, endH, beginW, endW);

                // Accumulate the log luminance
                if (L > eps)
                {
                  sum.first += math::log2(L);
                  sum.second++;
                }
              }
            }

            return sum;
          },
          [](Sum a, Sum b) -> Sum { return Sum(a.first+b.first, a.second+b.second); }
        );

      *dstPtr = (sum.second > 0) ? (key / math::exp2(sum.first / float(sum.second))) : 1.f;
    }, ct);
  }

OIDN_NAMESPACE_END