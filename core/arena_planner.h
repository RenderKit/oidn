// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common/platform.h"
#include <vector>

OIDN_NAMESPACE_BEGIN

// Plans allocations in an arena for a sequence of operations, determining the offsets of the
// allocations and the total required amount of memory
class ArenaPlanner final
{
public:
  // Adds an allocation as part of the specified operation, and returns the allocation ID
  int newAlloc(int opID, size_t byteSize, size_t byteAlignment = memoryAlignment);
  int newAlloc(int opID, SizeAndAlignment byteSizeAndAlignment);

  // Adds allocation dependencies for the specified operation, optionally requiring these
  // allocations to be stored consecutively in memory
  void addDepAllocs(int opID, const std::vector<int>& allocIDs, bool concatAllocs = false);

  // Commits changes to the plan, after which it's possible to query the offsets of the allocations
  void commit();

  // Clears the plan
  void clear();

  // Returns the total required memory size (must be called after committing)
  size_t getByteSize() const
  {
    checkCommitted();
    return totalByteSize;
  }

  // Returns the total required memory alignment (must be called after committing)
  size_t getByteAlignment() const
  {
    checkCommitted();
    return totalByteAlignment;
  }

  // Returns the offset of the specified allocation (must be called after committing)
  size_t getAllocByteOffset(int allocID) const
  {
    checkCommitted();
    checkAllocID(allocID);
    return allocs[allocID]->byteOffset;
  }

private:
  // Allocation record
  struct Alloc
  {
    size_t byteSize;      // size of the allocation in bytes
    size_t byteAlignment; // required alignment of the allocation in bytes
    size_t byteOffset;    // offset of the allocation in bytes (set later when committing)

    int firstOpID;        // index of the first operation that uses this allocation
    int lastOpID;         // index of the last operation that uses this allocation
    Alloc* next;          // allocation stored consecutively after this one
    Alloc* prev;          // allocation stored consecutively before this one

    Alloc(int opID, size_t byteSize, size_t byteAlignment)
      : byteSize(byteSize),
        byteAlignment(byteAlignment),
        byteOffset(0),
        firstOpID(opID),
        lastOpID(opID),
        next(nullptr),
        prev(nullptr) {}
  };

  void checkCommitted() const
  {
    if (dirty)
      throw std::logic_error("arena allocation plan is not committed");
  }

  void checkOpID(int opID) const
  {
    if (opID < 0)
      throw std::out_of_range("invalid operation ID");
  }

  void checkAllocID(int allocID) const
  {
    if (allocID < 0 || allocID >= static_cast<int>(allocs.size()))
      throw std::out_of_range("invalid arena allocation ID");
  }

  std::vector<std::unique_ptr<Alloc>> allocs;
  size_t totalByteSize = 0;
  size_t totalByteAlignment = 1;
  bool dirty = false;
};

OIDN_NAMESPACE_END