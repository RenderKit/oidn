// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_io.h"

OIDN_NAMESPACE_BEGIN

  ImageBuffer::ImageBuffer()
    : devPtr(nullptr),
      hostPtr(nullptr),
      numValues(0),
      width(0),
      height(0),
      numChannels(0),
      dataType(Format::Undefined) {}

  ImageBuffer::ImageBuffer(const DeviceRef& device, int width, int height, int numChannels, Format dataType, Storage storage)
    : device(device),
      numValues(size_t(width) * height * numChannels),
      width(width),
      height(height),
      numChannels(numChannels),
      dataType(dataType)
  {
    size_t valueByteSize = 0;
    switch (dataType)
    {
    case Format::Float:
      valueByteSize = sizeof(float);
      break;
    case Format::Half:
      valueByteSize = sizeof(int16_t);
      break;
    default:
      assert(0);
    }
  
    byteSize = std::max(numValues * valueByteSize, size_t(1)); // avoid zero-sized buffer
    buffer = (storage == Storage::Undefined) ? device.newBuffer(byteSize) : device.newBuffer(byteSize, storage);
    devPtr = static_cast<char*>(buffer.getData());
    hostPtr = (storage != Storage::Device) ? devPtr : nullptr;
  }

  void ImageBuffer::map(Access access)
  {
    assert(hostPtr == nullptr);
    hostPtr = static_cast<char*>(buffer.map(access));
  }

  void ImageBuffer::unmap()
  {
    assert(hostPtr);
    buffer.unmap(hostPtr);
    hostPtr = nullptr;
  }

  std::shared_ptr<ImageBuffer> ImageBuffer::clone() const
  {
    assert(hostPtr);
    auto result = std::make_shared<ImageBuffer>(device, width, height, numChannels, dataType);
    memcpy(result->hostPtr, hostPtr, getByteSize());
    return result;
  }

  std::tuple<size_t, double> compareImage(const ImageBuffer& image,
                                          const ImageBuffer& ref)
  {
    assert(ref.getDims() == image.getDims());    

    size_t numErrors = 0;
    double avgError  = 0; // SMAPE

    for (size_t i = 0; i < image.getSize(); ++i)
    {
      const double actual = image.get(i);
      const double expect = ref.get(i);

      const double absError = std::abs(expect - actual);
      const double relError = absError / (std::abs(expect) + std::abs(actual) + 0.01);

      if (absError > 0.01 && relError > 0.05)
      {
        if (numErrors < 5)
          std::cerr << "  error i=" << i << ", expect=" << expect << ", actual=" << actual << std::endl;
        ++numErrors;
      }

      avgError += relError;
    }

    avgError /= image.getSize();
    if (avgError > 0.05)
      numErrors = image.getSize();

    return std::make_tuple(numErrors, avgError);
  }

OIDN_NAMESPACE_END
