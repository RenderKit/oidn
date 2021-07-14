// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common/exception.h"
#include "tza.h"

namespace oidn {

  // Checks for buffer overrun
  __forceinline void checkBounds(char* ptr, char* end, size_t size)
  {
    if (end - ptr < (ptrdiff_t)size)
      throw Exception(Error::InvalidOperation, "invalid or corrupted weights blob");
  }

  // Reads a value from a buffer (with bounds checking) and advances the pointer
  template<typename T>
  __forceinline T read(char*& ptr, char* end)
  {
    checkBounds(ptr, end, sizeof(T));
    T value;
    memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
    return value;
  }

  std::map<std::string, std::shared_ptr<Tensor>> parseTZA(const Ref<Device>& device, void* buffer, size_t size)
  {
    char* input = (char*)buffer;
    char* const bufferEnd = input + size;

    // Parse the magic value
    const int magic = read<uint16_t>(input, bufferEnd);
    if (magic != 0x41D7)
      throw Exception(Error::InvalidOperation, "invalid or corrupted weights blob");

    // Parse the version
    const int majorVersion = read<uint8_t>(input, bufferEnd);
    const int minorVersion = read<uint8_t>(input, bufferEnd);
    UNUSED(minorVersion);
    if (majorVersion != 2)
      throw Exception(Error::InvalidOperation, "unsupported weights blob version");

    // Parse the table offset and jump to the table
    const uint64_t tableOffset = read<uint64_t>(input, bufferEnd);
    input = (char*)buffer + tableOffset;

    // Parse the number of tensors
    const size_t numTensors = read<uint32_t>(input, bufferEnd);

    // Parse the tensors
    std::map<std::string, std::shared_ptr<Tensor>> tensorMap;
    for (size_t i = 0; i < numTensors; ++i)
    {
      TensorDesc tensorDesc;

      // Parse the name
      const size_t nameLen = read<uint16_t>(input, bufferEnd);
      checkBounds(input, bufferEnd, nameLen);
      std::string name(input, nameLen);
      input += nameLen;

      // Parse the number of dimensions
      const int ndims = read<uint8_t>(input, bufferEnd);

      // Parse the shape of the tensor
      tensorDesc.dims.resize(ndims);
      for (int j = 0; j < ndims; ++j)
        tensorDesc.dims[j] = read<uint32_t>(input, bufferEnd);

      // Parse the layout of the tensor
      checkBounds(input, bufferEnd, ndims);
      std::string layout = std::string(input, input + ndims);
      if (layout == "x")
        tensorDesc.layout = TensorLayout::x;
      else if (layout == "oihw")
        tensorDesc.layout = TensorLayout::oihw;
      else
        throw Exception(Error::InvalidOperation, "invalid tensor layout");
      input += ndims;

      // Parse the data type of the tensor
      const char dataType = read<char>(input, bufferEnd);
      if (dataType == 'f')
        tensorDesc.dataType = DataType::Float32;
      else
        throw Exception(Error::InvalidOperation, "invalid tensor data type");

      // Parse the offset to the tensor data
      const uint64_t tensorOffset = read<uint64_t>(input, bufferEnd);
      char* tensorData = (char*)buffer + tensorOffset;
      checkBounds(tensorData, bufferEnd, tensorDesc.byteSize());

      // Add the tensor to the map
      auto tensor = std::make_shared<Tensor>(device, tensorDesc, tensorData);
      tensorMap.emplace(name, tensor);
    }

    return tensorMap;
  }

} // namespace oidn
