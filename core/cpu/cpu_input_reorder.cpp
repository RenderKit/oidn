// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_input_reorder.h"
#include "input_reorder_kernel_ispc.h"

namespace oidn {

  CPUInputReorderNode::CPUInputReorderNode(const Ref<CPUDevice>& device, const InputReorderDesc& desc)
    : CPUNode(device, desc.name),
      InputReorderNode(desc) {}

  void CPUInputReorderNode::execute()
  {
    assert(tile.H + tile.hSrcBegin <= getInput()->height);
    assert(tile.W + tile.wSrcBegin <= getInput()->width);
    assert(tile.H + tile.hDstBegin <= dst->height());
    assert(tile.W + tile.wDstBegin <= dst->width());

    ispc::InputReorder kernel;

    kernel.color  = color  ? *color  : Image();
    kernel.albedo = albedo ? *albedo : Image();
    kernel.normal = normal ? *normal : Image();
    kernel.dst = *dst;
    kernel.tile = tile;
    kernel.transferFunc = *transferFunc;
    kernel.hdr = hdr;
    kernel.snorm = snorm;

    parallel_nd(kernel.dst.H, [&](int hDst)
    {
      ispc::InputReorder_kernel(&kernel, hDst);
    });
  }

} // namespace oidn