// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "exception.h"
#include "tza.h"

namespace oidn {

  std::map<std::string, Tensor> parseTensors(void* buffer)
  {
    char* input = (char*)buffer;

    // Parse the magic value
    const int magic = *(unsigned short*)input;
    if (magic != 0x41D7)
      throw Exception(Error::InvalidOperation, "invalid tensor format");
    input += sizeof(unsigned short);

    // Parse the version
    const int majorVersion = *(unsigned char*)input++;
    const int minorVersion = *(unsigned char*)input++;
    UNUSED(minorVersion);
    if (majorVersion != 2)
      throw Exception(Error::InvalidOperation, "unsupported tensor format version");

    // Parse the table offset and jump to the table
    const uint64_t tableOffset = *(uint64_t*)input;
    input = (char*)buffer + tableOffset;

    // Parse the number of tensors
    const size_t numTensors = *(uint32_t*)input;
    input += sizeof(uint32_t);

    // Parse the tensors
    std::map<std::string, Tensor> tensorMap;
    for (size_t i = 0; i < numTensors; ++i)
    {
      Tensor tensor;

      // Parse the name
      const size_t nameLen = *(uint16_t*)input;
      input += sizeof(uint16_t);
      std::string name(input, nameLen);
      input += nameLen;

      // Parse the number of dimensions
      const int ndims = *(uint8_t*)input++;

      // Parse the shape of the tensor
      tensor.dims.resize(ndims);
      for (int j = 0; j < ndims; ++j)
        tensor.dims[j] = ((uint32_t*)input)[j];
      input += ndims * sizeof(uint32_t);

      // Parse the layout of the tensor
      tensor.layout = std::string(input, input + ndims);
      input += ndims;

      // Parse the data type of the tensor
      const char type = *(char*)input++;
      if (type != 'f') // only float32 is supported
        throw Exception(Error::InvalidOperation, "unsupported tensor data type");

      // Parse the offset to the tensor data
      const uint64_t offset = *(uint64_t*)input;
      input += sizeof(uint64_t);
      tensor.data = (float*)((char*)buffer + offset);

      // Add the tensor to the map
      tensorMap.emplace(name, std::move(tensor));
    }

    return tensorMap;
  }

} // namespace oidn
