// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

OIDN_NAMESPACE_BEGIN

  // -----------------------------------------------------------------------------------------------
  // WorkDim
  // -----------------------------------------------------------------------------------------------

  template<int N>
  class WorkDim
  {
  public:
    template<int n = N>
    OIDN_HOST_DEVICE_INLINE WorkDim(enable_if_t<n == 1, int> dim0)
      : dim{dim0} {}

    template<int n = N>
    OIDN_HOST_DEVICE_INLINE WorkDim(enable_if_t<n == 2, int> dim0, int dim1)
      : dim{dim0, dim1} {}

    template<int n = N>
    OIDN_HOST_DEVICE_INLINE WorkDim(enable_if_t<n == 3, int> dim0, int dim1, int dim2)
      : dim{dim0, dim1, dim2} {}

    template<int n = N>
    OIDN_HOST_DEVICE_INLINE operator enable_if_t<n == 1, int>() const { return dim[0]; }

    OIDN_HOST_DEVICE_INLINE const int& operator [](int i) const { return dim[i]; }
    OIDN_HOST_DEVICE_INLINE int& operator [](int i) { return dim[i]; }

  #if defined(OIDN_COMPILE_SYCL)
    template<int n = N>
    OIDN_INLINE operator enable_if_t<n == 1, sycl::range<1>>() const { return sycl::range<1>(dim[0]); }

    template<int n = N>
    OIDN_INLINE operator enable_if_t<n == 2, sycl::range<2>>() const { return sycl::range<2>(dim[0], dim[1]); }

    template<int n = N>
    OIDN_INLINE operator enable_if_t<n == 3, sycl::range<3>>() const { return sycl::range<3>(dim[0], dim[1], dim[2]); }
  #elif defined(OIDN_COMPILE_CUDA) || defined(OIDN_COMPILE_HIP)
    OIDN_INLINE operator dim3() const
    {
      if (N == 1)
        return dim3(dim[0]);
      else if (N == 2)
        return dim3(dim[1], dim[0]);
      else
        return dim3(dim[2], dim[1], dim[0]);
    }
  #endif

    OIDN_INLINE int getLinearSize() const
    {
      int size = dim[0];
      for (int i = 1; i < N; ++i)
        size *= dim[i];
      return size;
    }

  private:
    int dim[N];
  };

  OIDN_INLINE WorkDim<1> operator *(WorkDim<1> a, WorkDim<1> b) { return {a[0] * b[0]}; }
  OIDN_INLINE WorkDim<2> operator *(WorkDim<2> a, WorkDim<2> b) { return {a[0] * b[0], a[1] * b[1]}; }
  OIDN_INLINE WorkDim<3> operator *(WorkDim<3> a, WorkDim<3> b) { return {a[0] * b[0], a[1] * b[1], a[2] * b[2]}; }

  OIDN_INLINE WorkDim<1> operator /(WorkDim<1> a, WorkDim<1> b) { return {a[0] / b[0]}; }
  OIDN_INLINE WorkDim<2> operator /(WorkDim<2> a, WorkDim<2> b) { return {a[0] / b[0], a[1] / b[1]}; }
  OIDN_INLINE WorkDim<3> operator /(WorkDim<3> a, WorkDim<3> b) { return {a[0] / b[0], a[1] / b[1], a[2] / b[2]}; }

  OIDN_INLINE WorkDim<1> ceil_div(WorkDim<1> a, WorkDim<1> b) {
    return {ceil_div(a[0], b[0])};
  }
  OIDN_INLINE WorkDim<2> ceil_div(WorkDim<2> a, WorkDim<2> b) {
    return {ceil_div(a[0], b[0]), ceil_div(a[1], b[1])};
  }
  OIDN_INLINE WorkDim<3> ceil_div(WorkDim<3> a, WorkDim<3> b) {
    return {ceil_div(a[0], b[0]), ceil_div(a[1], b[1]), ceil_div(a[2], b[2])};
  }

#if defined(OIDN_COMPILE_SYCL)

  template<typename T>
  using GlobalPtr = sycl::multi_ptr<T, sycl::access::address_space::global_space>;

  // -----------------------------------------------------------------------------------------------
  // SYCL WorkItem
  // -----------------------------------------------------------------------------------------------

  template<int N>
  class WorkItem
  {
  public:
    OIDN_DEVICE_INLINE WorkItem(const sycl::item<N>& item) : item(item) {}

    template<int i = 0> OIDN_DEVICE_INLINE int getGlobalID()   const { return int(item.get_id(i)); }
    template<int i = 0> OIDN_DEVICE_INLINE int getGlobalSize() const { return int(item.get_range(i)); }

    OIDN_DEVICE_INLINE int getGlobalLinearID() const { return int(item.get_linear_id()); }

  private:
    const sycl::item<N>& item;
  };

  // -----------------------------------------------------------------------------------------------
  // SYCL WorkGroupItem
  // -----------------------------------------------------------------------------------------------

  template<int N>
  class WorkGroupItem
  {
  public:
    OIDN_DEVICE_INLINE WorkGroupItem(const sycl::nd_item<N>& item) : item(item) {}

    template<int i = 0> OIDN_DEVICE_INLINE int getGlobalID()   const { return int(item.get_global_id(i)); }
    template<int i = 0> OIDN_DEVICE_INLINE int getGlobalSize() const { return int(item.get_global_range(i)); }
    template<int i = 0> OIDN_DEVICE_INLINE int getLocalID()    const { return int(item.get_local_id(i)); }
    template<int i = 0> OIDN_DEVICE_INLINE int getLocalSize()  const { return int(item.get_local_range(i)); }
    template<int i = 0> OIDN_DEVICE_INLINE int getGroupID()    const { return int(item.get_group(i)); }
    template<int i = 0> OIDN_DEVICE_INLINE int getNumGroups()  const { return int(item.get_group_range(i)); }

    OIDN_DEVICE_INLINE int getGlobalLinearID() const { return int(item.get_global_linear_id()); }
    OIDN_DEVICE_INLINE int getLocalLinearID()  const { return int(item.get_local_linear_id()); }
    OIDN_DEVICE_INLINE int getGroupLinearID()  const { return int(item.get_group_linear_id()); }

    OIDN_DEVICE_INLINE void groupBarrier() const
    {
      item.barrier(sycl::access::fence_space::local_space);
    }

    OIDN_DEVICE_INLINE int getSubgroupLocalID() const
    {
      return int(item.get_sub_group().get_local_linear_id());
    }

    OIDN_DEVICE_INLINE int getSubgroupSize() const
    {
      return int(item.get_sub_group().get_max_local_range()[0]);
    }

    template<typename T>
    OIDN_DEVICE_INLINE T subgroupBroadcast(T x, int id) const
    {
      return sycl::group_broadcast(item.get_sub_group(), x, id);
    }

    template<typename T>
    OIDN_DEVICE_INLINE T subgroupShuffle(T x, int id) const
    {
      return sycl::select_from_group(item.get_sub_group(), x, id);
    }

    template<typename T>
    OIDN_DEVICE_INLINE void subgroupStore(GlobalPtr<T> dst, const T& x) const
    {
      item.get_sub_group().store(dst, x);
    }

    OIDN_DEVICE_INLINE void subgroupBarrier() const
    {
      sycl::group_barrier(item.get_sub_group());
    }

  private:
    const sycl::nd_item<N>& item;
  };

  // -----------------------------------------------------------------------------------------------
  // SYCL WorkGroup
  // -----------------------------------------------------------------------------------------------

  template<int N>
  struct WorkGroup
  {
    // Shared local array
    template<typename T, int size>
    class LocalArray
    {
    public:
      OIDN_DEVICE_INLINE const T& operator [](unsigned int i) const { return (*ptr)[i]; }
      OIDN_DEVICE_INLINE T& operator [](unsigned int i) { return (*ptr)[i]; }

    private:
      sycl::multi_ptr<T[size], sycl::access::address_space::local_space> ptr =
        sycl::ext::oneapi::group_local_memory<T[size]>(sycl::ext::oneapi::experimental::this_nd_item<N>().get_group());
    };
  };

#elif defined(OIDN_COMPILE_CUDA) || defined(OIDN_COMPILE_HIP)

  template<typename T>
  using GlobalPtr = T*;

  // -----------------------------------------------------------------------------------------------
  // CUDA/HIP WorkItem
  // -----------------------------------------------------------------------------------------------

  template<int N>
  class WorkItem
  {
  public:
    template<int n = N>
    OIDN_DEVICE_INLINE WorkItem(typename std::enable_if<n == 1, const WorkDim<1>&>::type globalSize)
      : globalID(blockIdx.x * blockDim.x + threadIdx.x),
        globalSize(globalSize) {}

    template<int n = N>
    OIDN_DEVICE_INLINE WorkItem(typename std::enable_if<n == 2, const WorkDim<2>&>::type globalSize)
      : globalID(blockIdx.y * blockDim.y + threadIdx.y,
                 blockIdx.x * blockDim.x + threadIdx.x),
        globalSize(globalSize) {}

    template<int n = N>
    OIDN_DEVICE_INLINE WorkItem(typename std::enable_if<n == 3, const WorkDim<3>&>::type globalSize)
      : globalID(blockIdx.z * blockDim.z + threadIdx.z,
                 blockIdx.y * blockDim.y + threadIdx.y,
                 blockIdx.x * blockDim.x + threadIdx.x),
        globalSize(globalSize) {}

    template<int i = 0> OIDN_DEVICE_INLINE int getGlobalID()   const { return globalID[i]; }
    template<int i = 0> OIDN_DEVICE_INLINE int getGlobalSize() const { return globalSize[i]; }

  private:
    WorkDim<N> globalID;
    WorkDim<N> globalSize;
  };

  // -----------------------------------------------------------------------------------------------
  // CUDA/HIP WorkGroupItem
  // -----------------------------------------------------------------------------------------------

  class WorkGroupItemBase
  {
  public:
    OIDN_DEVICE_INLINE void groupBarrier() const { __syncthreads(); }

    OIDN_DEVICE_INLINE int getSubgroupSize() const { return warpSize; }

    template<typename T>
    OIDN_DEVICE_INLINE T subgroupShuffle(T x, int id) const
    {
    #if defined(OIDN_COMPILE_CUDA)
      return __shfl_sync(0xFFFFFFFF, x, id);
    #else
      return __shfl(x, id); // HIP doesn't support __shfl_sync
    #endif
    }

    template<typename T>
    OIDN_DEVICE_INLINE T subgroupBroadcast(T x, int id) const { return subgroupShuffle(x, id); }

  #if defined(OIDN_COMPILE_CUDA)
    OIDN_DEVICE_INLINE void subgroupBarrier() const { __syncwarp(); }
  #endif
  };

  template<int N>
  class WorkGroupItem;

  template<>
  class WorkGroupItem<1> : public WorkGroupItemBase
  {
  public:
    OIDN_DEVICE_INLINE int getGlobalID()   const { return blockIdx.x * blockDim.x + threadIdx.x; }
    OIDN_DEVICE_INLINE int getGlobalSize() const { return gridDim.x * blockDim.x; }
    OIDN_DEVICE_INLINE int getLocalID()    const { return threadIdx.x; };
    OIDN_DEVICE_INLINE int getLocalSize()  const { return blockDim.x; };
    OIDN_DEVICE_INLINE int getGroupID()    const { return blockIdx.x; };
    OIDN_DEVICE_INLINE int getNumGroups()  const { return gridDim.x; }

    OIDN_DEVICE_INLINE int getSubgroupLocalID() const { return threadIdx.x % warpSize; }
  };

  template<>
  class WorkGroupItem<2> : public WorkGroupItemBase
  {
  public:
    template<int i> OIDN_DEVICE_INLINE int getGlobalID()   const;
    template<int i> OIDN_DEVICE_INLINE int getGlobalSize() const;
    template<int i> OIDN_DEVICE_INLINE int getLocalID()    const;
    template<int i> OIDN_DEVICE_INLINE int getLocalSize()  const;
    template<int i> OIDN_DEVICE_INLINE int getGroupID()    const;
    template<int i> OIDN_DEVICE_INLINE int getNumGroups()  const;

    OIDN_DEVICE_INLINE int getGlobalLinearID()  const;
    OIDN_DEVICE_INLINE int getLocalLinearID()   const { return threadIdx.y * blockDim.x + threadIdx.x; };
    OIDN_DEVICE_INLINE int getGroupLinearID()   const { return blockIdx.y * gridDim.x + blockIdx.x; }
    OIDN_DEVICE_INLINE int getSubgroupLocalID() const { return getLocalLinearID() % warpSize; }
  };

  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGlobalID<0>() const { return blockIdx.y * blockDim.y + threadIdx.y; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGlobalID<1>() const { return blockIdx.x * blockDim.x + threadIdx.x; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGlobalSize<0>() const { return gridDim.y * blockDim.y; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGlobalSize<1>() const { return gridDim.x * blockDim.x; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getLocalID<0>() const { return threadIdx.y; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getLocalID<1>() const { return threadIdx.x; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getLocalSize<0>() const { return blockDim.y; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getLocalSize<1>() const { return blockDim.x; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGroupID<0>() const { return blockIdx.y; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGroupID<1>() const { return blockIdx.x; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getNumGroups<0>() const { return gridDim.y; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getNumGroups<1>() const { return gridDim.x; }

  OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGlobalLinearID() const
  {
    return getGlobalID<0>() * getGlobalSize<1>() + getGlobalID<1>();
  };

  // -----------------------------------------------------------------------------------------------
  // CUDA/HIP WorkGroup
  // -----------------------------------------------------------------------------------------------

  template<int N>
  struct WorkGroup
  {
    // Shared local array, must be declared with OIDN_SHARED
    template<typename T, int size>
    class LocalArray
    {
    public:
      OIDN_DEVICE_INLINE const T& operator [](unsigned int i) const { return v[i]; }
      OIDN_DEVICE_INLINE T& operator [](unsigned int i) { return v[i]; }

    private:
      T v[size];
    };
  };

#endif

OIDN_NAMESPACE_END