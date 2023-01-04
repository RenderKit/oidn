// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_io.h"

namespace oidn {

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
  
    size_t byteSize = std::max(numValues * valueByteSize, size_t(1)); // avoid zero-sized buffer
    buffer = (storage == Storage::Undefined) ? device.newBuffer(byteSize) : device.newBuffer(byteSize, storage);
    devPtr = (char*)buffer.getData();
    hostPtr = (storage != Storage::Device) ? devPtr : nullptr;
  }

  void ImageBuffer::map(Access access)
  {
    assert(hostPtr == nullptr);
    hostPtr = (char*)buffer.map(access);
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

  std::tuple<size_t, float> compareImage(const ImageBuffer& image,
                                         const ImageBuffer& ref,
                                         float threshold)
  {
    assert(ref.getDims() == image.getDims());    

    size_t numErrors = 0;
    float maxError = 0;

    for (size_t i = 0; i < image.getSize(); ++i)
    {
      const float actual = image.get(i);
      const float expect = ref.get(i);

      float error = std::abs(expect - actual);
      if (expect != 0)
        error = std::min(error, error / expect);

      maxError = std::max(maxError, error);
      if (error > threshold)
      {
        //std::cerr << "i=" << i << " expect=" << expect << " actual=" << actual;
        ++numErrors;
      }
    }

    return std::make_tuple(numErrors, maxError);
  }

} // namespace oidn
