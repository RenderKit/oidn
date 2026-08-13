// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tza.h"
#include "exception.h"

OIDN_NAMESPACE_BEGIN

  // Checks whether the range [ptr, ptr + size) is inside the buffer [begin, end). Both ends of the
  // buffer must be checked because an offset large enough to wrap the pointer around lands *below*
  // the buffer, where checking only the end would find plenty of space remaining.
  oidn_inline void checkBounds(const char* ptr, size_t size, const char* begin, const char* end)
  {
    // Comparing sizes rather than casting to ptrdiff_t, which would make sizes above PTRDIFF_MAX
    // negative and thus silently pass the check
    if (ptr < begin || ptr > end || size > size_t(end - ptr))
      throw Exception(Error::InvalidOperation, "invalid or corrupted weights blob");
  }

  // Reads a value from a buffer (with bounds checking) and advances the pointer
  template<typename T>
  oidn_inline T read(const char*& ptr, const char* begin, const char* end)
  {
    checkBounds(ptr, sizeof(T), begin, end);
    T value;
    memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
    return value;
  }

  std::shared_ptr<TensorMap> parseTZA(const void* buffer, size_t size)
  {
    const char* const bufferBegin = static_cast<const char*>(buffer);
    const char* const bufferEnd   = bufferBegin + size;
    const char* input = bufferBegin;

    // Parse the magic value
    const int magic = read<uint16_t>(input, bufferBegin, bufferEnd);
    if (magic != 0x41D7)
      throw Exception(Error::InvalidOperation, "invalid or corrupted weights blob");

    // Parse the version
    const int majorVersion = read<uint8_t>(input, bufferBegin, bufferEnd);
    const int minorVersion = read<uint8_t>(input, bufferBegin, bufferEnd);
    UNUSED(minorVersion);
    if (majorVersion != 2)
      throw Exception(Error::InvalidOperation, "unsupported weights blob version");

    // Parse the table offset and jump to the table. The offset must be validated against the size
    // of the blob *before* it is added to the base pointer, because a large enough offset would
    // wrap the pointer around to below the blob, and merely forming such a pointer is already
    // undefined behavior, regardless of the bounds check performed on it afterwards.
    const uint64_t tableOffset = read<uint64_t>(input, bufferBegin, bufferEnd);
    if (!isRangeValid(tableOffset, sizeof(uint32_t), size))
      throw Exception(Error::InvalidOperation, "invalid or corrupted weights blob");
    input = bufferBegin + tableOffset;

    // Parse the number of tensors
    const size_t numTensors = read<uint32_t>(input, bufferBegin, bufferEnd);

    // Parse the tensors
    std::shared_ptr<TensorMap> tensorMap = std::make_shared<TensorMap>();
    for (size_t i = 0; i < numTensors; ++i)
    {
      TensorDesc tensorDesc;

      // Parse the name
      const size_t nameLen = read<uint16_t>(input, bufferBegin, bufferEnd);
      checkBounds(input, nameLen, bufferBegin, bufferEnd);
      std::string name(input, nameLen);
      input += nameLen;

      // Parse the number of dimensions
      const int ndims = read<uint8_t>(input, bufferBegin, bufferEnd);

      // Parse the shape of the tensor. The dimensions are limited to the largest power of two
      // which can be represented as a signed int. Using INT_MAX instead would result in an
      // overflow when rounding up to the nearest multiple of the block size.
      constexpr uint32_t maxTensorDim = uint32_t(1) << 30;

      tensorDesc.dims.resize(ndims);
      for (int j = 0; j < ndims; ++j)
      {
        const uint32_t dim = read<uint32_t>(input, bufferBegin, bufferEnd);
        if (dim == 0 || dim > maxTensorDim)
          throw Exception(Error::InvalidOperation, "invalid tensor dimension");
        tensorDesc.dims[j] = int(dim);
      }
      tensorDesc.paddedDims = tensorDesc.dims;

      // Parse the layout of the tensor
      checkBounds(input, ndims, bufferBegin, bufferEnd);
      std::string layout = std::string(input, input + ndims);
      if (layout == "x")
        tensorDesc.layout = TensorLayout::x;
      else if (layout == "oihw")
        tensorDesc.layout = TensorLayout::oihw;
      else
        throw Exception(Error::InvalidOperation, "invalid tensor layout");
      input += ndims;

      // Parse the data type of the tensor
      const char dataType = read<char>(input, bufferBegin, bufferEnd);
      if (dataType == 'f')
        tensorDesc.dataType = DataType::Float32;
      else if (dataType == 'h')
        tensorDesc.dataType = DataType::Float16;
      else
        throw Exception(Error::InvalidOperation, "invalid tensor data type");

      // Make sure that getByteSize() below cannot overflow, which would wrap the size of the
      // tensor to a small value and let the bounds check pass for a tensor larger than the blob
      size_t numElements = 1;
      for (int dim : tensorDesc.paddedDims)
      {
        if (!isMulSafe(numElements, size_t(dim)))
          throw Exception(Error::InvalidOperation, "invalid or corrupted weights blob");
        numElements *= size_t(dim);
      }
      if (!isMulSafe(numElements, getDataTypeSize(tensorDesc.dataType)))
        throw Exception(Error::InvalidOperation, "invalid or corrupted weights blob");

      // Parse the offset to the tensor data, validating it against the size of the blob before
      // adding it to the base pointer (see the table offset above)
      const uint64_t tensorOffset = read<uint64_t>(input, bufferBegin, bufferEnd);
      if (!isRangeValid(tensorOffset, tensorDesc.getByteSize(), size))
        throw Exception(Error::InvalidOperation, "invalid or corrupted weights blob");
      const char* tensorData = bufferBegin + tensorOffset;

      // Add the tensor to the map
      auto tensor = makeRef<HostTensor>(tensorDesc, const_cast<char*>(tensorData));
      tensorMap->emplace(name, tensor);
    }

    return tensorMap;
  }

OIDN_NAMESPACE_END
