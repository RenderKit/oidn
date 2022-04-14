// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(OIDN_HIP)
  #include <hip/hip_runtime.h>
#endif

namespace oidn {

  // ---------------------------------------------------------------------------
  // WorkRange
  // ---------------------------------------------------------------------------

  template<int dims>
  class WorkRange;

  template<>
  class WorkRange<1>
  {
  public:
    WorkRange(int range0) : r{range0} {}
    const int& operator [](size_t i) const { assert(i < 1); return r; }
    int& operator [](size_t i) { assert(i < 1); return r; }
    operator int() const { return r; }

  private:
    int r;
  };

  template<>
  class WorkRange<2>
  {
  public:
    WorkRange(int range0, int range1) : r{range0, range1} {}
    const int& operator [](size_t i) const { assert(i < 2); return r[i]; }
    int& operator [](size_t i) { assert(i < 2); return r[i]; }

  private:
    int r[2];
  };

  template<>
  class WorkRange<3>
  {
  public:
    WorkRange(int range0, int range1, int range2) : r{range0, range1, range2} {}
    const int& operator [](size_t i) const { assert(i < 3); return r[i]; }
    int& operator [](size_t i) { assert(i < 3); return r[i]; }

  private:
    int r[3];
  };

#if defined(OIDN_SYCL)

  // ---------------------------------------------------------------------------
  // SYCL WorkItem
  // ---------------------------------------------------------------------------

  template<int dims>
  class WorkItem
  {
  public:
    WorkItem(const sycl::item<dims>& item) : item(item) {}

    template<int i = 0>
    OIDN_DEVICE_INLINE int getId() const
    {
      return int(item.get_id(i));
    }

    OIDN_DEVICE_INLINE int getLinearId() const
    {
      return int(item.get_linear_id());
    }

    template<int i = 0>
    OIDN_DEVICE_INLINE int getRange() const
    {
      return int(item.get_range(i));
    }

  private:
    const sycl::item<dims>& item;
  };

  // ---------------------------------------------------------------------------
  // SYCL WorkGroupItem
  // ---------------------------------------------------------------------------

  template<int dims>
  class WorkGroupItem
  {
  public:
    WorkGroupItem(const sycl::nd_item<dims>& item) : item(item) {}

    template<int i = 0> OIDN_DEVICE_INLINE int getGlobalId() const { return int(item.get_global_id(i)); }
    template<int i = 0> OIDN_DEVICE_INLINE int getGlobalRange() const { return int(item.get_global_range(i)); }
    template<int i = 0> OIDN_DEVICE_INLINE int getLocalId() const { return int(item.get_local_id(i)); }
    template<int i = 0> OIDN_DEVICE_INLINE int getLocalRange() const { return int(item.get_local_range(i)); }
    template<int i = 0> OIDN_DEVICE_INLINE int getGroupId() const { return int(item.get_group(i)); }
    template<int i = 0> OIDN_DEVICE_INLINE int getGroupRange() const { return int(item.get_group_range(i)); }

    OIDN_DEVICE_INLINE int getGlobalLinearId() const { return int(item.get_global_linear_id()); };
    OIDN_DEVICE_INLINE int getLocalLinearId() const { return int(item.get_local_linear_id()); };
    OIDN_DEVICE_INLINE int getGroupLinearId() const { return int(item.get_group_linear_id()); }

    OIDN_DEVICE_INLINE void syncGroup() const
    {
      item.barrier(sycl::access::fence_space::local_space);
    }

  private:
    const sycl::nd_item<dims>& item;
  };

  // ---------------------------------------------------------------------------
  // SYCL WorkGroup
  // ---------------------------------------------------------------------------

  template<int dims>
  struct WorkGroup
  {
    // Shared local array
    template<typename T, int N>
    class LocalArray
    {
    public:
      OIDN_DEVICE_INLINE const T& operator [](unsigned int i) const { return (*ptr)[i]; }
      OIDN_DEVICE_INLINE T& operator [](unsigned int i) { return (*ptr)[i]; }

    private:
      sycl::multi_ptr<T[N], sycl::access::address_space::local_space> ptr =
        sycl::ext::oneapi::group_local_memory<T[N]>(sycl::ext::oneapi::experimental::this_nd_item<dims>().get_group());
    };
  };

#elif defined(OIDN_CUDA) || defined(OIDN_HIP)

  // ---------------------------------------------------------------------------
  // CUDA/HIP WorkItem
  // ---------------------------------------------------------------------------

  template<int dims>
  class WorkItem;

  template<>
  class WorkItem<1>
  {
  public:
    OIDN_DEVICE_INLINE WorkItem(int range)
      : id(blockIdx.x * blockDim.x + threadIdx.x),
        range(range) {}

    OIDN_DEVICE_INLINE int getId() const { return id; }
    OIDN_DEVICE_INLINE int getRange() const { return range; }

  private:
    int id;
    int range;
  };

  template<>
  class WorkItem<2>
  {
  public:
    OIDN_DEVICE_INLINE WorkItem(int range0, int range1)
      : id0(blockIdx.y * blockDim.y + threadIdx.y),
        id1(blockIdx.x * blockDim.x + threadIdx.x),
        range0(range0),
        range1(range1) {}

    template<int i> OIDN_DEVICE_INLINE int getId() const;
    template<int i> OIDN_DEVICE_INLINE int getRange() const;

  private:
    int id0, id1;
    int range0, range1;
  };

  template<> OIDN_DEVICE_INLINE int WorkItem<2>::getId<0>() const { return id0; }
  template<> OIDN_DEVICE_INLINE int WorkItem<2>::getId<1>() const { return id1; }
  template<> OIDN_DEVICE_INLINE int WorkItem<2>::getRange<0>() const { return range0; }
  template<> OIDN_DEVICE_INLINE int WorkItem<2>::getRange<1>() const { return range1; }

  template<>
  class WorkItem<3>
  {
  public:
    OIDN_DEVICE_INLINE WorkItem(int range0, int range1, int range2)
      : id0(blockIdx.z * blockDim.z + threadIdx.z),
        id1(blockIdx.y * blockDim.y + threadIdx.y),
        id2(blockIdx.x * blockDim.x + threadIdx.x),
        range0(range0),
        range1(range1),
        range2(range2) {}

    template<int i> OIDN_DEVICE_INLINE int getId() const;
    template<int i> OIDN_DEVICE_INLINE int getRange() const;

  private:
    int id0, id1, id2;
    int range0, range1, range2;
  };

  template<> OIDN_DEVICE_INLINE int WorkItem<3>::getId<0>() const { return id0; }
  template<> OIDN_DEVICE_INLINE int WorkItem<3>::getId<1>() const { return id1; }
  template<> OIDN_DEVICE_INLINE int WorkItem<3>::getId<2>() const { return id2; }
  template<> OIDN_DEVICE_INLINE int WorkItem<3>::getRange<0>() const { return range0; }
  template<> OIDN_DEVICE_INLINE int WorkItem<3>::getRange<1>() const { return range1; }
  template<> OIDN_DEVICE_INLINE int WorkItem<3>::getRange<2>() const { return range2; }

  // ---------------------------------------------------------------------------
  // CUDA/HIP WorkGroupItem
  // ---------------------------------------------------------------------------

  template<int dims>
  class WorkGroupItem;

  template<>
  class WorkGroupItem<1>
  {
  public:
    OIDN_DEVICE_INLINE int getGlobalId() const { return blockIdx.x * blockDim.x + threadIdx.x; }
    OIDN_DEVICE_INLINE int getGlobalRange() const { return gridDim.x * blockDim.x; }
    OIDN_DEVICE_INLINE int getLocalId() const { return threadIdx.x; };
    OIDN_DEVICE_INLINE int getLocalRange() const { return blockDim.x; };
    OIDN_DEVICE_INLINE int getGroupId() const { return blockIdx.x; };
    OIDN_DEVICE_INLINE int getGroupRange() const { return gridDim.x; }

    OIDN_DEVICE_INLINE void syncGroup() const { __syncthreads(); }
  };

  template<>
  class WorkGroupItem<2>
  {
  public:
    template<int i> OIDN_DEVICE_INLINE int getGlobalId() const;
    template<int i> OIDN_DEVICE_INLINE int getGlobalRange() const;
    template<int i> OIDN_DEVICE_INLINE int getLocalId() const;
    template<int i> OIDN_DEVICE_INLINE int getLocalRange() const;
    template<int i> OIDN_DEVICE_INLINE int getGroupId() const;
    template<int i> OIDN_DEVICE_INLINE int getGroupRange() const;

    OIDN_DEVICE_INLINE int getGlobalLinearId() const;
    OIDN_DEVICE_INLINE int getLocalLinearId() const { return threadIdx.y * blockDim.x + threadIdx.x; };
    OIDN_DEVICE_INLINE int getGroupLinearId() const { return blockIdx.y * gridDim.x + blockIdx.x; }
    
    OIDN_DEVICE_INLINE void syncGroup() const { __syncthreads(); }
  };

  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGlobalId<0>() const { return blockIdx.y * blockDim.y + threadIdx.y; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGlobalId<1>() const { return blockIdx.x * blockDim.x + threadIdx.x; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGlobalRange<0>() const { return gridDim.y * blockDim.y; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGlobalRange<1>() const { return gridDim.x * blockDim.x; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getLocalId<0>() const { return threadIdx.y; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getLocalId<1>() const { return threadIdx.x; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getLocalRange<0>() const { return blockDim.y; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getLocalRange<1>() const { return blockDim.x; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGroupId<0>() const { return blockIdx.y; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGroupId<1>() const { return blockIdx.x; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGroupRange<0>() const { return gridDim.y; }
  template<> OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGroupRange<1>() const { return gridDim.x; }

  OIDN_DEVICE_INLINE int WorkGroupItem<2>::getGlobalLinearId() const
  {
    return getGlobalId<0>() * getGlobalRange<1>() + getGlobalId<1>();
  };

  // ---------------------------------------------------------------------------
  // CUDA/HIP WorkGroup
  // ---------------------------------------------------------------------------

  template<int dims>
  struct WorkGroup
  {
    // Shared local array, must be declared with OIDN_SHARED
    template<typename T, int N>
    class LocalArray
    {
    public:
      OIDN_DEVICE_INLINE const T& operator [](unsigned int i) const { return v[i]; }
      OIDN_DEVICE_INLINE T& operator [](unsigned int i) { return v[i]; }

    private:
      T v[N];
    };
  };

#endif

} // namespace oidn