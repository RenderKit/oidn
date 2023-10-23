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
    OIDN_HOST_DEVICE_INLINE WorkDim(int dim0) : dim{dim0} {}
    OIDN_HOST_DEVICE_INLINE operator int() const { return dim[0]; }
    OIDN_HOST_DEVICE_INLINE const oidn_private int& operator [](int i) const { return dim[i]; }
    OIDN_HOST_DEVICE_INLINE oidn_private int& operator [](int i) { return dim[i]; }

  #if defined(OIDN_COMPILE_SYCL)
    OIDN_INLINE operator sycl::range<1>() const { return sycl::range<1>(dim[0]); }
  #elif defined(OIDN_COMPILE_CUDA) || defined(OIDN_COMPILE_HIP)
    OIDN_INLINE operator dim3() const { return dim3(dim[0]); }
  #elif defined(OIDN_COMPILE_METAL_HOST)
    OIDN_INLINE operator MTLSize() const { return MTLSizeMake(dim[0], 1, 1); }
  #endif

    OIDN_INLINE int getLinearSize() const { return dim[0]; }

  private:
    int dim[1];
  };

  template<>
  class WorkDim<2>
  {
  public:
    OIDN_HOST_DEVICE_INLINE WorkDim(int dim0, int dim1) : dim{dim0, dim1} {}
    OIDN_HOST_DEVICE_INLINE const oidn_private int& operator [](int i) const { return dim[i]; }
    OIDN_HOST_DEVICE_INLINE oidn_private int& operator [](int i) { return dim[i]; }

  #if defined(OIDN_COMPILE_SYCL)
    OIDN_INLINE operator sycl::range<2>() const { return sycl::range<2>(dim[0], dim[1]); }
  #elif defined(OIDN_COMPILE_CUDA) || defined(OIDN_COMPILE_HIP)
    OIDN_INLINE operator dim3() const { return dim3(dim[1], dim[0]); }
  #elif defined(OIDN_COMPILE_METAL_HOST)
    OIDN_INLINE operator MTLSize() const { return MTLSizeMake(dim[1], dim[0], 1); }
  #endif

    OIDN_INLINE int getLinearSize() const { return dim[0] * dim[1]; }

  private:
    int dim[2];
  };

  template<>
  class WorkDim<3>
  {
  public:
    OIDN_HOST_DEVICE_INLINE WorkDim(int dim0, int dim1, int dim2) : dim{dim0, dim1, dim2} {}
    OIDN_HOST_DEVICE_INLINE const oidn_private int& operator [](int i) const { return dim[i]; }
    OIDN_HOST_DEVICE_INLINE oidn_private int& operator [](int i) { return dim[i]; }

  #if defined(OIDN_COMPILE_SYCL)
    OIDN_INLINE operator sycl::range<3>() const { return sycl::range<3>(dim[0], dim[1], dim[2]); }
  #elif defined(OIDN_COMPILE_CUDA) || defined(OIDN_COMPILE_HIP)
    OIDN_INLINE operator dim3() const { return dim3(dim[2], dim[1], dim[0]); }
  #elif defined(OIDN_COMPILE_METAL_HOST)
    OIDN_INLINE operator MTLSize() const { return MTLSizeMake(dim[2], dim[1], dim[0]); }
  #endif

    OIDN_INLINE int getLinearSize() const { return dim[0] * dim[1] * dim[2]; }

  private:
    int dim[3];
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

  // -----------------------------------------------------------------------------------------------
  // WorkItem, WorkGroupItem
  // -----------------------------------------------------------------------------------------------

#if defined(OIDN_COMPILE_SYCL)

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

#elif defined(OIDN_COMPILE_CUDA) || defined(OIDN_COMPILE_HIP)

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

#elif defined(OIDN_COMPILE_METAL_DEVICE)

  template<int N>
  class WorkItem;

  template<>
  class WorkItem<2>
  {
  public:
    WorkItem(uint2 globalID, uint2 globalSize)
      : globalID(globalID), globalSize(globalSize) {}

    template<int i> OIDN_DEVICE_INLINE int getGlobalID() const   { return globalID[1-i]; }
    template<int i> OIDN_DEVICE_INLINE int getGlobalSize() const { return globalSize[1-i]; }

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

    OIDN_DEVICE_INLINE int getGlobalID()   const { return globalID; }
    OIDN_DEVICE_INLINE int getGlobalSize() const { return globalSize; }
    OIDN_DEVICE_INLINE int getLocalID()    const { return localID; }
    OIDN_DEVICE_INLINE int getLocalSize()  const { return localSize; }
    OIDN_DEVICE_INLINE int getGroupID()    const { return groupID; }
    OIDN_DEVICE_INLINE int getNumGroups()  const { return numGroups; }

    OIDN_DEVICE_INLINE int getGlobalLinearID() const { return globalID; }
    OIDN_DEVICE_INLINE int getLocalLinearID()  const { return localID; }
    OIDN_DEVICE_INLINE int getGroupLinearID()  const { return groupID; }

    OIDN_DEVICE_INLINE void groupBarrier() const
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

    template<int i> OIDN_DEVICE_INLINE int getGlobalID()   const { return globalID[1-i];   }
    template<int i> OIDN_DEVICE_INLINE int getGlobalSize() const { return globalSize[1-i]; }
    template<int i> OIDN_DEVICE_INLINE int getLocalID()    const { return localID[1-i];    }
    template<int i> OIDN_DEVICE_INLINE int getLocalSize()  const { return localSize[1-i];  }
    template<int i> OIDN_DEVICE_INLINE int getGroupID()    const { return groupID[1-i];    }
    template<int i> OIDN_DEVICE_INLINE int getNumGroups()  const { return numGroups[1-i];  }

    OIDN_DEVICE_INLINE int getGlobalLinearID() const { return globalID[1] * globalSize[0] + globalID[0]; }
    OIDN_DEVICE_INLINE int getLocalLinearID()  const { return localID[1]  * localSize[0]  + localID[0];  }
    OIDN_DEVICE_INLINE int getGroupLinearID()  const { return groupID[1]  * numGroups[0]  + groupID[0];  }

    OIDN_DEVICE_INLINE void groupBarrier() const
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
    template<int i = 0> OIDN_INLINE int getGlobalID()   const { return globalID[i]; }
    template<int i = 0> OIDN_INLINE int getGlobalSize() const { return globalSize[i]; }

    OIDN_INLINE int getGlobalLinearID() const
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
    template<int i = 0> OIDN_INLINE int getGlobalID()   const { return globalID[i]; }
    template<int i = 0> OIDN_INLINE int getGlobalSize() const { return globalSize[i]; }
    template<int i = 0> OIDN_INLINE int getLocalID()    const { return localID[i]; }
    template<int i = 0> OIDN_INLINE int getLocalSize()  const { return localSize[i]; }
    template<int i = 0> OIDN_INLINE int getGroupID()    const { return groupID[i]; }
    template<int i = 0> OIDN_INLINE int getNumGroups()  const { return numGroups[i]; }

    OIDN_INLINE int getGlobalLinearID() const
    {
      int id = 0;
      for (int i = 0; i < N; ++i)
        id = id * globalSize[i] + globalID[i];
      return id;
    }

    OIDN_INLINE int getLocalLinearID() const
    {
      int id = 0;
      for (int i = 0; i < N; ++i)
        id = id * localSize[i] + localID[i];
      return id;
    }

    OIDN_INLINE int getGroupLinearID() const
    {
      int id = 0;
      for (int i = 0; i < N; ++i)
        id = id * numGroups[i] + groupID[i];
      return id;
    }

    OIDN_INLINE void groupBarrier() const {}

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