// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "arena_planner.h"

OIDN_NAMESPACE_BEGIN

  int ArenaPlanner::newAlloc(int opID, size_t byteSize, size_t byteAlignment)
  {
    checkOpID(opID);

    const int allocID = static_cast<int>(allocs.size());
    allocs.emplace_back(new Alloc(opID, byteSize, byteAlignment));
    dirty = true;
    return allocID;
  }

  int ArenaPlanner::newAlloc(int opID, SizeAndAlignment byteSizeAndAlignment)
  {
    return newAlloc(opID, byteSizeAndAlignment.size, byteSizeAndAlignment.alignment);
  }

  void ArenaPlanner::addDepAllocs(int opID, const std::vector<int>& allocIDs, bool concatAllocs)
  {
    checkOpID(opID);
    for (int allocID : allocIDs)
      checkAllocID(allocID);

    Alloc* prev = nullptr;
    for (int allocID : allocIDs)
    {
      Alloc* cur = allocs[allocID].get();

      if (opID < cur->firstOpID)
        throw std::logic_error("arena allocation cannot be used before it is created");
      cur->lastOpID = max(cur->lastOpID, opID);

      if (concatAllocs && prev)
      {
        if (cur->prev || prev->next)
          throw std::logic_error("invalid arena allocation planning constraints");
        cur->prev = prev;
        prev->next = cur;
      }

      prev = cur;
    }

    dirty = true;
  }

  void ArenaPlanner::commit()
  {
    if (!dirty)
      return;

    // Determine the chunks to allocate. Each chunk contains one or more allocations consecutively
    struct Chunk
    {
      Alloc* firstAlloc;
      int firstOpID;
      int lastOpID;
      size_t byteSize;
      size_t byteAlignment;
    };

    std::vector<Chunk> chunks;

    // Iterate over all allocations and find the first allocation in each chunk
    for (const auto& alloc : allocs)
    {
      // If the allocation is not the first in a chunk, skip it
      if (alloc->prev)
        continue;

      // Initialize the chunk
      Chunk chunk;
      chunk.firstAlloc    = alloc.get();
      chunk.byteSize      = 0;
      chunk.byteAlignment = alloc->byteAlignment;
      chunk.firstOpID     = alloc->firstOpID;
      chunk.lastOpID      = alloc->lastOpID;

      // Iterate over all allocations in the chunk
      for (Alloc* curAlloc = chunk.firstAlloc; curAlloc; curAlloc = curAlloc->next)
      {
        chunk.byteSize += curAlloc->byteSize;
        chunk.firstOpID = min(chunk.firstOpID, curAlloc->firstOpID);
        chunk.lastOpID  = max(chunk.lastOpID,  curAlloc->lastOpID);
      }

      chunks.push_back(chunk);
    }

    // Sort the chunks by size in descending order
    std::sort(chunks.begin(), chunks.end(),
              [](const Chunk& a, const Chunk& b) { return a.byteSize > b.byteSize; });

    // Track the active allocations sorted by offset in ascending order
    std::vector<Alloc*> activeAllocs;
    totalByteSize = 0;

    // Iterate over the sorted chunks to allocate
    for (const Chunk& chunk : chunks)
    {
      size_t curByteOffset   = 0;
      size_t bestByteOffset  = SIZE_MAX;
      size_t bestGapByteSize = SIZE_MAX;

      // Iterate over the active allocations sorted by offset in ascending order
      // Find the smallest gap between them that is large enough to fit the chunk
      for (const Alloc* alloc : activeAllocs)
      {
        // If the allocation does not overlap with the chunk in time, skip it
        if (alloc->lastOpID < chunk.firstOpID || alloc->firstOpID > chunk.lastOpID)
          continue;

        const size_t curAlignedByteOffset = round_up(curByteOffset, chunk.byteAlignment);

        // Check whether the current gap is large enough to fit the chunk and
        // is smaller than the previous best fit
        if (curAlignedByteOffset + chunk.byteSize <= alloc->byteOffset &&
            alloc->byteOffset - curByteOffset < bestGapByteSize)
        {
          bestByteOffset  = curAlignedByteOffset;
          bestGapByteSize = alloc->byteOffset - curByteOffset;
        }

        curByteOffset = max(curByteOffset, alloc->byteOffset + alloc->byteSize);
      }

      if (bestByteOffset == SIZE_MAX)
        bestByteOffset = round_up(curByteOffset, chunk.byteAlignment);

      // Assign offsets to the allocations in the chunk, and add them to the sorted active allocations
      for (Alloc* alloc = chunk.firstAlloc; alloc; alloc = alloc->next)
      {
        alloc->byteOffset = bestByteOffset;

        auto it = std::upper_bound(activeAllocs.begin(), activeAllocs.end(), alloc,
                    [](const Alloc* a, const Alloc* b) { return a->byteOffset < b->byteOffset; });
        activeAllocs.insert(it, alloc);

        bestByteOffset += alloc->byteSize;
      }

      totalByteSize = max(totalByteSize, bestByteOffset);
      totalByteAlignment = lcm(totalByteAlignment, chunk.byteAlignment);
    }

    dirty = false;
  }

  void ArenaPlanner::clear()
  {
    allocs.clear();
    totalByteSize = 0;
    totalByteAlignment = 1;
    dirty = false;
  }

OIDN_NAMESPACE_END