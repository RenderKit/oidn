// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "color.h"

namespace oidn {

  TransferFunction::TransferFunction(Type type)
    : type(type)
  {
  }

  TransferFunction::operator ispc::TransferFunction() const
  {
    ispc::TransferFunction res;

    switch (type)
    {
    case Type::Linear: ispc::LinearTransferFunction_Constructor(&res); break;
    case Type::SRGB:   ispc::SRGBTransferFunction_Constructor(&res);   break;
    case Type::PU:     ispc::PUTransferFunction_Constructor(&res);     break;
    case Type::Log:    ispc::LogTransferFunction_Constructor(&res);    break;
    default:
      assert(0);
    }

    res.inputScale  = inputScale;
    res.outputScale = outputScale;
    
    return res;
  }

  float getAutoexposure(const Image& color)
  {
    constexpr float key = 0.18f;
    constexpr float eps = 1e-8f;
    constexpr int K = 16; // downsampling amount

    // Downsample the image to minimize sensitivity to noise
    const int H  = color.height;  // original height
    const int W  = color.width;   // original width
    const int HK = (H + K/2) / K; // downsampled height
    const int WK = (W + K/2) / K; // downsampled width

    ispc::ImageAccessor colorIspc = color;

    // Compute the average log luminance of the downsampled image
    using Sum = std::pair<float, int>;

    Sum sum =
      tbb::parallel_reduce(
        tbb::blocked_range2d<int>(0, HK, 0, WK),
        Sum(0.f, 0),
        [&](const tbb::blocked_range2d<int>& r, Sum sum) -> Sum
        {
          // Iterate over blocks
          for (int i = r.rows().begin(); i != r.rows().end(); ++i)
          {
            for (int j = r.cols().begin(); j != r.cols().end(); ++j)
            {
              // Compute the average luminance in the current block
              const int beginH = int(ptrdiff_t(i)   * H / HK);
              const int beginW = int(ptrdiff_t(j)   * W / WK);
              const int endH   = int(ptrdiff_t(i+1) * H / HK);
              const int endW   = int(ptrdiff_t(j+1) * W / WK);

              const float L = ispc::getAvgLuminance(colorIspc, beginH, endH, beginW, endW);

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

    return (sum.second > 0) ? (key / exp2(sum.first / float(sum.second))) : 1.f;
  }

} // namespace oidn
