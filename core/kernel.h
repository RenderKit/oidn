// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"
#if defined(OIDN_COMPILE_METAL_HOST)
  #include <Metal/Metal.h>
#endif

OIDN_NAMESPACE_BEGIN

  // -----------------------------------------------------------------------------------------------
  // GlobalPtr, LocalPtr
  // -----------------------------------------------------------------------------------------------

#if defined(OIDN_COMPILE_SYCL)

  template<typename T>
  using GlobalPtr = sycl::multi_ptr<T, sycl::access::address_space::global_space>;

  template<typename T>
  using LocalPtr  = sycl::multi_ptr<T, sycl::access::address_space::local_space>;

#else
  template<typename T>
  using GlobalPtr = oidn_global T*;

  template<typename T>
  using LocalPtr  = oidn_local T*;
#endif

  // -----------------------------------------------------------------------------------------------
  // WorkDim
  // -----------------------------------------------------------------------------------------------

  template<int N>
  class WorkDim;

  template<>
  class WorkDim<1>
  {
  public:
    oidn_host_device_inline WorkDim(int dim0) : dim{dim0} {}
    oidn_host_device_inline operator int() const { return dim[0]; }
    oidn_host_device_inline const oidn_private int& operator [](int i) const { return dim[i]; }
    oidn_host_device_inline oidn_private int& operator [](int i) { return dim[i]; }

  #if defined(OIDN_COMPILE_SYCL)
    oidn_inline operator sycl::range<1>() const { return sycl::range<1>(dim[0]); }
  #elif defined(OIDN_COMPILE_CUDA) || defined(OIDN_COMPILE_HIP)
    oidn_inline operator dim3() const { return dim3(dim[0]); }
  #elif defined(OIDN_COMPILE_METAL_HOST)
    oidn_inline operator MTLSize() const { return MTLSizeMake(dim[0], 1, 1); }
  #endif

    oidn_inline int getLinearSize() const { return dim[0]; }

  private:
    int dim[1];
  };

  template<>
  class WorkDim<2>
  {
  public:
    oidn_host_device_inline WorkDim(int dim0, int dim1) : dim{dim0, dim1} {}
    oidn_host_device_inline const oidn_private int& operator [](int i) const { return dim[i]; }
    oidn_host_device_inline oidn_private int& operator [](int i) { return dim[i]; }

  #if defined(OIDN_COMPILE_SYCL)
    oidn_inline operator sycl::range<2>() const { return sycl::range<2>(dim[0], dim[1]); }
  #elif defined(OIDN_COMPILE_CUDA) || defined(OIDN_COMPILE_HIP)
    oidn_inline operator dim3() const { return dim3(dim[1], dim[0]); }
  #elif defined(OIDN_COMPILE_METAL_HOST)
    oidn_inline operator MTLSize() const { return MTLSizeMake(dim[1], dim[0], 1); }
  #endif

    oidn_inline int getLinearSize() const { return dim[0] * dim[1]; }

  private:
    int dim[2];
  };

  template<>
  class WorkDim<3>
  {
  public:
    oidn_host_device_inline WorkDim(int dim0, int dim1, int dim2) : dim{dim0, dim1, dim2} {}
    oidn_host_device_inline const oidn_private int& operator [](int i) const { return dim[i]; }
    oidn_host_device_inline oidn_private int& operator [](int i) { return dim[i]; }

  #if defined(OIDN_COMPILE_SYCL)
    oidn_inline operator sycl::range<3>() const { return sycl::range<3>(dim[0], dim[1], dim[2]); }
  #elif defined(OIDN_COMPILE_CUDA) || defined(OIDN_COMPILE_HIP)
    oidn_inline operator dim3() const { return dim3(dim[2], dim[1], dim[0]); }
  #elif defined(OIDN_COMPILE_METAL_HOST)
    oidn_inline operator MTLSize() const { return MTLSizeMake(dim[2], dim[1], dim[0]); }
  #endif

    oidn_inline int getLinearSize() const { return dim[0] * dim[1] * dim[2]; }

  private:
    int dim[3];
  };

  oidn_inline WorkDim<1> operator *(WorkDim<1> a, WorkDim<1> b) { return {a[0] * b[0]}; }
  oidn_inline WorkDim<2> operator *(WorkDim<2> a, WorkDim<2> b) { return {a[0] * b[0], a[1] * b[1]}; }
  oidn_inline WorkDim<3> operator *(WorkDim<3> a, WorkDim<3> b) { return {a[0] * b[0], a[1] * b[1], a[2] * b[2]}; }

  oidn_inline WorkDim<1> operator /(WorkDim<1> a, WorkDim<1> b) { return {a[0] / b[0]}; }
  oidn_inline WorkDim<2> operator /(WorkDim<2> a, WorkDim<2> b) { return {a[0] / b[0], a[1] / b[1]}; }
  oidn_inline WorkDim<3> operator /(WorkDim<3> a, WorkDim<3> b) { return {a[0] / b[0], a[1] / b[1], a[2] / b[2]}; }

  oidn_inline WorkDim<1> ceil_div(WorkDim<1> a, WorkDim<1> b) {
    return {ceil_div(a[0], b[0])};
  }
  oidn_inline WorkDim<2> ceil_div(WorkDim<2> a, WorkDim<2> b) {
    return {ceil_div(a[0], b[0]), ceil_div(a[1], b[1])};
  }
  oidn_inline WorkDim<3> ceil_div(WorkDim<3> a, WorkDim<3> b) {
    return {ceil_div(a[0], b[0]), ceil_div(a[1], b[1]), ceil_div(a[2], b[2])};
  }

  // -----------------------------------------------------------------------------------------------
  // WorkItem, WorkGroupItem
  // -----------------------------------------------------------------------------------------------

#if defined(OIDN_COMPILE_SYCL)

  template<int N>
  class WorkItem
  {
  public:
    oidn_device_inline WorkItem(const sycl::item<N>& item) : item(item) {}

    template<int i = 0> oidn_device_inline int getGlobalID()   const { return int(item.get_id(i)); }
    template<int i = 0> oidn_device_inline int getGlobalSize() const { return int(item.get_range(i)); }

    oidn_device_inline int getGlobalLinearID() const { return int(item.get_linear_id()); }

  private:
    const sycl::item<N>& item;
  };

  template<int N>
  class WorkGroupItem
  {
  public:
    oidn_device_inline WorkGroupItem(const sycl::nd_item<N>& item) : item(item) {}

    template<int i = 0> oidn_device_inline int getGlobalID()   const { return int(item.get_global_id(i)); }
    template<int i = 0> oidn_device_inline int getGlobalSize() const { return int(item.get_global_range(i)); }
    template<int i = 0> oidn_device_inline int getLocalID()    const { return int(item.get_local_id(i)); }
    template<int i = 0> oidn_device_inline int getLocalSize()  const { return int(item.get_local_range(i)); }
    template<int i = 0> oidn_device_inline int getGroupID()    const { return int(item.get_group(i)); }
    template<int i = 0> oidn_device_inline int getNumGroups()  const { return int(item.get_group_range(i)); }

    oidn_device_inline int getGlobalLinearID() const { return int(item.get_global_linear_id()); }
    oidn_device_inline int getLocalLinearID()  const { return int(item.get_local_linear_id()); }
    oidn_device_inline int getGroupLinearID()  const { return int(item.get_group_linear_id()); }

    oidn_device_inline void groupBarrier() const
    {
      item.barrier(sycl::access::fence_space::local_space);
    }

    oidn_device_inline int getSubgroupLocalID() const
    {
      return int(item.get_sub_group().get_local_linear_id());
    }

    oidn_device_inline int getSubgroupSize() const
    {
      return int(item.get_sub_group().get_max_local_range()[0]);
    }

    template<typename T>
    oidn_device_inline T subgroupBroadcast(T x, int id) const
    {
      return sycl::group_broadcast(item.get_sub_group(), x, id);
    }

    template<typename T>
    oidn_device_inline T subgroupShuffle(T x, int id) const
    {
      return sycl::select_from_group(item.get_sub_group(), x, id);
    }

    template<typename T>
    oidn_device_inline void subgroupStore(GlobalPtr<T> dst, const T& x) const
    {
      item.get_sub_group().store(dst, x);
    }

    oidn_device_inline void subgroupBarrier() const
    {
      sycl::group_barrier(item.get_sub_group());
    }

  private:
    const sycl::nd_item<N>& item;
  };

#elif defined(OIDN_COMPILE_CUDA) || defined(OIDN_COMPILE_HIP)

  template<int N>
  class WorkItem
  {
  public:
    template<int n = N>
    oidn_device_inline WorkItem(typename std::enable_if<n == 1, const WorkDim<1>&>::type globalSize)
      : globalID(blockIdx.x * blockDim.x + threadIdx.x),
        globalSize(globalSize) {}

    template<int n = N>
    oidn_device_inline WorkItem(typename std::enable_if<n == 2, const WorkDim<2>&>::type globalSize)
      : globalID(blockIdx.y * blockDim.y + threadIdx.y,
                 blockIdx.x * blockDim.x + threadIdx.x),
        globalSize(globalSize) {}

    template<int n = N>
    oidn_device_inline WorkItem(typename std::enable_if<n == 3, const WorkDim<3>&>::type globalSize)
      : globalID(blockIdx.z * blockDim.z + threadIdx.z,
                 blockIdx.y * blockDim.y + threadIdx.y,
                 blockIdx.x * blockDim.x + threadIdx.x),
        globalSize(globalSize) {}

    template<int i = 0> oidn_device_inline int getGlobalID()   const { return globalID[i]; }
    template<int i = 0> oidn_device_inline int getGlobalSize() const { return globalSize[i]; }

  private:
    WorkDim<N> globalID;
    WorkDim<N> globalSize;
  };

  class WorkGroupItemBase
  {
  public:
    oidn_device_inline void groupBarrier() const { __syncthreads(); }

    oidn_device_inline int getSubgroupSize() const { return warpSize; }

    template<typename T>
    oidn_device_inline T subgroupShuffle(T x, int id) const
    {
    #if defined(OIDN_COMPILE_CUDA)
      return __shfl_sync(0xFFFFFFFF, x, id);
    #else
      return __shfl(x, id); // HIP doesn't support __shfl_sync
    #endif
    }

    template<typename T>
    oidn_device_inline T subgroupBroadcast(T x, int id) const { return subgroupShuffle(x, id); }

  #if defined(OIDN_COMPILE_CUDA)
    oidn_device_inline void subgroupBarrier() const { __syncwarp(); }
  #endif
  };

  template<int N>
  class WorkGroupItem;

  template<>
  class WorkGroupItem<1> : public WorkGroupItemBase
  {
  public:
    oidn_device_inline int getGlobalID()   const { return blockIdx.x * blockDim.x + threadIdx.x; }
    oidn_device_inline int getGlobalSize() const { return gridDim.x * blockDim.x; }
    oidn_device_inline int getLocalID()    const { return threadIdx.x; };
    oidn_device_inline int getLocalSize()  const { return blockDim.x; };
    oidn_device_inline int getGroupID()    const { return blockIdx.x; };
    oidn_device_inline int getNumGroups()  const { return gridDim.x; }

    oidn_device_inline int getSubgroupLocalID() const { return threadIdx.x % warpSize; }
  };

  template<>
  class WorkGroupItem<2> : public WorkGroupItemBase
  {
  public:
    template<int i> oidn_device_inline int getGlobalID()   const;
    template<int i> oidn_device_inline int getGlobalSize() const;
    template<int i> oidn_device_inline int getLocalID()    const;
    template<int i> oidn_device_inline int getLocalSize()  const;
    template<int i> oidn_device_inline int getGroupID()    const;
    template<int i> oidn_device_inline int getNumGroups()  const;

    oidn_device_inline int getGlobalLinearID()  const;
    oidn_device_inline int getLocalLinearID()   const { return threadIdx.y * blockDim.x + threadIdx.x; };
    oidn_device_inline int getGroupLinearID()   const { return blockIdx.y * gridDim.x + blockIdx.x; }
    oidn_device_inline int getSubgroupLocalID() const { return getLocalLinearID() % warpSize; }
  };

  template<> oidn_device_inline int WorkGroupItem<2>::getGlobalID<0>() const { return blockIdx.y * blockDim.y + threadIdx.y; }
  template<> oidn_device_inline int WorkGroupItem<2>::getGlobalID<1>() const { return blockIdx.x * blockDim.x + threadIdx.x; }
  template<> oidn_device_inline int WorkGroupItem<2>::getGlobalSize<0>() const { return gridDim.y * blockDim.y; }
  template<> oidn_device_inline int WorkGroupItem<2>::getGlobalSize<1>() const { return gridDim.x * blockDim.x; }
  template<> oidn_device_inline int WorkGroupItem<2>::getLocalID<0>() const { return threadIdx.y; }
  template<> oidn_device_inline int WorkGroupItem<2>::getLocalID<1>() const { return threadIdx.x; }
  template<> oidn_device_inline int WorkGroupItem<2>::getLocalSize<0>() const { return blockDim.y; }
  template<> oidn_device_inline int WorkGroupItem<2>::getLocalSize<1>() const { return blockDim.x; }
  template<> oidn_device_inline int WorkGroupItem<2>::getGroupID<0>() const { return blockIdx.y; }
  template<> oidn_device_inline int WorkGroupItem<2>::getGroupID<1>() const { return blockIdx.x; }
  template<> oidn_device_inline int WorkGroupItem<2>::getNumGroups<0>() const { return gridDim.y; }
  template<> oidn_device_inline int WorkGroupItem<2>::getNumGroups<1>() const { return gridDim.x; }

  oidn_device_inline int WorkGroupItem<2>::getGlobalLinearID() const
  {
    return getGlobalID<0>() * getGlobalSize<1>() + getGlobalID<1>();
  };

#elif defined(OIDN_COMPILE_METAL_DEVICE)

  template<int N>
  class WorkItem;

  template<>
  class WorkItem<2>
  {
  public:
    WorkItem(uint2 globalID, uint2 globalSize)
      : globalID(globalID), globalSize(globalSize) {}

    template<int i> oidn_device_inline int getGlobalID() const   { return globalID[1-i]; }
    template<int i> oidn_device_inline int getGlobalSize() const { return globalSize[1-i]; }

  private:
    uint2 globalID, globalSize; // reverse indexing!
  };

  template<int N>
  class WorkGroupItem;

  template<>
  class WorkGroupItem<1>
  {
  public:
    WorkGroupItem(uint globalID, uint globalSize, uint localID, uint localSize,
                  uint groupID, uint numGroups)
      : globalID(globalID), globalSize(globalSize), localID(localID), localSize(localSize),
        groupID(groupID), numGroups(numGroups) {}

    oidn_device_inline int getGlobalID()   const { return globalID; }
    oidn_device_inline int getGlobalSize() const { return globalSize; }
    oidn_device_inline int getLocalID()    const { return localID; }
    oidn_device_inline int getLocalSize()  const { return localSize; }
    oidn_device_inline int getGroupID()    const { return groupID; }
    oidn_device_inline int getNumGroups()  const { return numGroups; }

    oidn_device_inline int getGlobalLinearID() const { return globalID; }
    oidn_device_inline int getLocalLinearID()  const { return localID; }
    oidn_device_inline int getGroupLinearID()  const { return groupID; }

    oidn_device_inline void groupBarrier() const
    {
      metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

  private:
    uint globalID, globalSize, localID, localSize, groupID, numGroups;
  };

  template<>
  class WorkGroupItem<2>
  {
  public:
    WorkGroupItem(uint2 globalID, uint2 globalSize, uint2 localID, uint2 localSize,
                  uint2 groupID, uint2 numGroups)
      : globalID(globalID), globalSize(globalSize), localID(localID), localSize(localSize),
        groupID(groupID), numGroups(numGroups) {}

    template<int i> oidn_device_inline int getGlobalID()   const { return globalID[1-i];   }
    template<int i> oidn_device_inline int getGlobalSize() const { return globalSize[1-i]; }
    template<int i> oidn_device_inline int getLocalID()    const { return localID[1-i];    }
    template<int i> oidn_device_inline int getLocalSize()  const { return localSize[1-i];  }
    template<int i> oidn_device_inline int getGroupID()    const { return groupID[1-i];    }
    template<int i> oidn_device_inline int getNumGroups()  const { return numGroups[1-i];  }

    oidn_device_inline int getGlobalLinearID() const { return globalID[1] * globalSize[0] + globalID[0]; }
    oidn_device_inline int getLocalLinearID()  const { return localID[1]  * localSize[0]  + localID[0];  }
    oidn_device_inline int getGroupLinearID()  const { return groupID[1]  * numGroups[0]  + groupID[0];  }

    oidn_device_inline void groupBarrier() const
    {
      metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

  private:
    uint2 globalID, globalSize, localID, localSize, groupID, numGroups; // reverse indexing!
  };

#else

  template<int N>
  class WorkItem
  {
  public:
    template<int i = 0> oidn_inline int getGlobalID()   const { return globalID[i]; }
    template<int i = 0> oidn_inline int getGlobalSize() const { return globalSize[i]; }

    oidn_inline int getGlobalLinearID() const
    {
      int id = 0;
      for (int i = 0; i < N; ++i)
        id = id * globalSize[i] + globalID[i];
      return id;
    }

  private:
    int globalID[N];
    int globalSize[N];
  };

  template<int N>
  class WorkGroupItem
  {
  public:
    template<int i = 0> oidn_inline int getGlobalID()   const { return globalID[i]; }
    template<int i = 0> oidn_inline int getGlobalSize() const { return globalSize[i]; }
    template<int i = 0> oidn_inline int getLocalID()    const { return localID[i]; }
    template<int i = 0> oidn_inline int getLocalSize()  const { return localSize[i]; }
    template<int i = 0> oidn_inline int getGroupID()    const { return groupID[i]; }
    template<int i = 0> oidn_inline int getNumGroups()  const { return numGroups[i]; }

    oidn_inline int getGlobalLinearID() const
    {
      int id = 0;
      for (int i = 0; i < N; ++i)
        id = id * globalSize[i] + globalID[i];
      return id;
    }

    oidn_inline int getLocalLinearID() const
    {
      int id = 0;
      for (int i = 0; i < N; ++i)
        id = id * localSize[i] + localID[i];
      return id;
    }

    oidn_inline int getGroupLinearID() const
    {
      int id = 0;
      for (int i = 0; i < N; ++i)
        id = id * numGroups[i] + groupID[i];
      return id;
    }

    oidn_inline void groupBarrier() const {}

  private:
    int globalID[N];
    int globalSize[N];
    int localID[N];
    int localSize[N];
    int groupID[N];
    int numGroups[N];
  };

#endif

OIDN_NAMESPACE_END