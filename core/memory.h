// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"

namespace oidn {

  inline memory::dims getMemoryDims(const std::shared_ptr<memory>& mem)
  {
    const dnnl_memory_desc_t& desc = mem->get_desc().data;
    return memory::dims(&desc.dims[0], &desc.dims[desc.ndims]);
  }

  inline memory::data_type getMemoryType(const std::shared_ptr<memory>& mem)
  {
    const dnnl_memory_desc_t& desc = mem->get_desc().data;
    return memory::data_type(desc.data_type);
  }

  // Returns the number of values in a memory object
  inline size_t getMemorySize(const memory::dims& dims)
  {
    size_t res = 1;
    for (int i = 0; i < (int)dims.size(); ++i)
      res *= dims[i];
    return res;
  }

  inline memory::dims getMaxMemoryDims(const std::vector<memory::dims>& dims)
  {
    memory::dims result;
    size_t maxSize = 0;

    for (const auto& d : dims)
    {
      const size_t size = getMemorySize(d);
      if (size > maxSize)
      {
        result = d;
        maxSize = size;
      }
    }

    return result;
  }

  inline size_t getMemorySize(const std::shared_ptr<memory>& mem)
  {
    return getMemorySize(getMemoryDims(mem));
  }

  inline ispc::Memory toIspc(const std::shared_ptr<memory>& mem)
  {
    const dnnl_memory_desc_t& desc = mem->get_desc().data;
    assert(desc.ndims == 4);
    assert(desc.dims[0] == 1);
    assert(desc.data_type == memory::data_type::f32);

    ispc::Memory res;
    res.ptr = (float*)mem->get_data_handle();
    res.C = desc.dims[1];
    res.H = desc.dims[2];
    res.W = desc.dims[3];
    return res;
  }

} // namespace oidn
