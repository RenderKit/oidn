// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_io.h"

OIDN_NAMESPACE_BEGIN

  ImageBuffer::ImageBuffer()
    : devPtr(nullptr),
      hostPtr(nullptr),
      byteSize(0),
      numValues(0),
      width(0),
      height(0),
      numChannels(0),
      dataType(DataType::Void),
      format(Format::Undefined) {}

  ImageBuffer::ImageBuffer(const DeviceRef& device, int width, int height, int numChannels,
                           DataType dataType, Storage storage, bool forceHostCopy)
    : device(device),
      numValues(size_t(width) * height * numChannels),
      width(width),
      height(height),
      numChannels(numChannels),
      dataType(dataType),
      format(makeFormat(dataType, numChannels))
  {
    const size_t valueByteSize = getDataTypeSize(dataType);
    byteSize = std::max(numValues * valueByteSize, size_t(1)); // avoid zero-sized buffer
    buffer = device.newBuffer(byteSize, storage);
    storage = buffer.getStorage(); // get actual storage mode
    devPtr  = (storage != Storage::Device) ? static_cast<char*>(buffer.getData()) : nullptr;
    hostPtr = (storage != Storage::Device && !forceHostCopy) ? devPtr : static_cast<char*>(malloc(byteSize));
  }

  ImageBuffer::~ImageBuffer()
  {
    if (hostPtr != devPtr)
      free(hostPtr);
  }

  void ImageBuffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr) const
  {
    buffer.read(byteOffset, byteSize, dstHostPtr);
  }

  void ImageBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr)
  {
    buffer.write(byteOffset, byteSize, srcHostPtr);
  }

  void ImageBuffer::toHost()
  {
    if (hostPtr != devPtr)
      buffer.read(0, byteSize, hostPtr);
  }

  void ImageBuffer::toHostAsync()
  {
    if (hostPtr != devPtr)
      buffer.readAsync(0, byteSize, hostPtr);
  }

  void ImageBuffer::toDevice()
  {
    if (hostPtr != devPtr)
      buffer.write(0, byteSize, hostPtr);
  }

  void ImageBuffer::toDeviceAsync()
  {
    if (hostPtr != devPtr)
      buffer.writeAsync(0, byteSize, hostPtr);
  }

  std::shared_ptr<ImageBuffer> ImageBuffer::clone() const
  {
    auto result = std::make_shared<ImageBuffer>(device, width, height, numChannels, dataType);
    buffer.read(0, byteSize, result->getHostData());
    return result;
  }

  bool compareImage(const ImageBuffer& image, const ImageBuffer& ref)
  {
    assert(ref.getDims() == image.getDims());

    for (size_t i = 0; i < image.getSize(); ++i)
    {
      const double actual = image.get(i);
      const double expect = ref.get(i);

      if (actual != expect)
        return false;
    }

    return true;
  }

  std::tuple<size_t, double> compareImage(const ImageBuffer& image,
                                          const ImageBuffer& ref,
                                          double errorThreshold)
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

      // Detect severe outliers
      if (!(absError <= 0.02 || relError <= 0.05) || (errorThreshold == 0 && actual != expect))
      {
        if (numErrors < 5)
          std::cerr << "  error i=" << i << ", expect=" << expect << ", actual=" << actual << std::endl;
        ++numErrors;
      }

      avgError += relError;
    }

    avgError /= image.getSize();

    if (!(avgError <= errorThreshold))
      numErrors = image.getSize();

    return std::make_tuple(numErrors, avgError);
  }

OIDN_NAMESPACE_END
