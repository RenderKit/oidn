// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(OIDN_HIP)
  #include <hip/hip_runtime.h>
#endif
#include "vec.h"

namespace oidn {

#if defined(OIDN_SYCL)

  // Shared local array
  template<typename T, int N, int dims>
  class KernelLocalArray
  {
  public:
    OIDN_DEVICE_INLINE T operator [](unsigned int i) const { return (*ptr)[i]; }
    OIDN_DEVICE_INLINE T& operator [](unsigned int i) { return (*ptr)[i]; }

  private:
    sycl::multi_ptr<T[N], sycl::access::address_space::local_space> ptr =
      sycl::ext::oneapi::group_local_memory<T[N]>(sycl::ext::oneapi::experimental::this_nd_item<dims>().get_group());
  };

  template<int dims>
  struct Kernel
  {
    template<typename T, int N>
    using LocalArray = KernelLocalArray<T, N, dims>;

    template<int i = 0>
    static OIDN_DEVICE_INLINE int getGlobalId()
    {
      return int(sycl::ext::oneapi::experimental::this_nd_item<dims>().get_global_id(i));
    }
    
    template<int i = 0>
    static OIDN_DEVICE_INLINE int getGlobalRange()
    {
      return int(sycl::ext::oneapi::experimental::this_nd_item<dims>().get_global_range(i));
    }

    template<int i = 0>
    static OIDN_DEVICE_INLINE int getLocalId()
    {
      return int(sycl::ext::oneapi::experimental::this_nd_item<dims>().get_local_id(i));
    }

    template<int i = 0>
    static OIDN_DEVICE_INLINE int getGroupId()
    {
      return int(sycl::ext::oneapi::experimental::this_nd_item<dims>().get_group(i));
    }

    template<int i = 0>
    static OIDN_DEVICE_INLINE int getGroupRange()
    {
      return int(sycl::ext::oneapi::experimental::this_nd_item<dims>().get_group_range(i));
    }

    static OIDN_DEVICE_INLINE int getLocalLinearId()
    {
      return sycl::ext::oneapi::experimental::this_nd_item<dims>().get_local_linear_id();
    };

    static OIDN_DEVICE_INLINE int getGroupLinearId()
    {
      return sycl::ext::oneapi::experimental::this_nd_item<dims>().get_group_linear_id();
    }

    static OIDN_DEVICE_INLINE void syncGroup()
    {
      sycl::ext::oneapi::experimental::this_nd_item<dims>().barrier(sycl::access::fence_space::local_space);
    }
  };

#elif defined(OIDN_CUDA) || defined(OIDN_HIP)

  // Shared local array, must be declared with OIDN_SHARED
  template<typename T, int N>
  class KernelLocalArray
  {
  public:
    OIDN_DEVICE_INLINE T operator [](unsigned int i) const { return v[i]; }
    OIDN_DEVICE_INLINE T& operator [](unsigned int i) { return v[i]; }

  private:
    T v[N];
  };

  template<int dims>
  struct Kernel;

  template<>
  struct Kernel<1>
  {
    template<typename T, int N>
    using LocalArray = KernelLocalArray<T, N>;

    static OIDN_DEVICE_INLINE int getGlobalId() { return blockIdx.x * blockDim.x + threadIdx.x; }
    static OIDN_DEVICE_INLINE int getGlobalRange() { return blockDim.x * gridDim.x; }
    static OIDN_DEVICE_INLINE int getLocalId() { return threadIdx.x; };
    static OIDN_DEVICE_INLINE int getGroupId() { return blockIdx.x; };
    static OIDN_DEVICE_INLINE int getGroupRange() { return gridDim.x; }

    static OIDN_DEVICE_INLINE void syncGroup() { __syncthreads(); }
  };

  template<>
  struct Kernel<2>
  {
    template<typename T, int N>
    using LocalArray = KernelLocalArray<T, N>;

    template<int i> static OIDN_DEVICE_INLINE int getLocalId();
    template<int i> static OIDN_DEVICE_INLINE int getGroupId();
    template<int i> static OIDN_DEVICE_INLINE int getGroupRange();
    static OIDN_DEVICE_INLINE int getLocalLinearId() { return threadIdx.y * blockDim.x + threadIdx.x; };
    static OIDN_DEVICE_INLINE int getGroupLinearId() { return blockIdx.y * gridDim.x + blockIdx.x; }
    
    static OIDN_DEVICE_INLINE void syncGroup() { __syncthreads(); }
  };

  template<> OIDN_DEVICE_INLINE int Kernel<2>::getLocalId<0>() { return threadIdx.y; }
  template<> OIDN_DEVICE_INLINE int Kernel<2>::getLocalId<1>() { return threadIdx.x; }
  template<> OIDN_DEVICE_INLINE int Kernel<2>::getGroupId<0>() { return blockIdx.y; }
  template<> OIDN_DEVICE_INLINE int Kernel<2>::getGroupId<1>() { return blockIdx.x; }
  template<> OIDN_DEVICE_INLINE int Kernel<2>::getGroupRange<0>() { return gridDim.y; }
  template<> OIDN_DEVICE_INLINE int Kernel<2>::getGroupRange<1>() { return gridDim.x; }

#endif

} // namespace oidn