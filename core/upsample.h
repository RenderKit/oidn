// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"

namespace oidn {

  // 2x2 nearest-neighbor upsampling node (blocked layout)
  class UpsampleNode : public Node
  {
  protected:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;

  public:
    UpsampleNode(const Ref<Device>& device,
                 const std::string& name,
                 const std::shared_ptr<Tensor>& src,
                 const std::shared_ptr<Tensor>& dst)
      : Node(device, name),
        src(src),
        dst(dst)
    {
      assert(src->ndims() == 3);
      assert(dst->ndims() == 3);
      assert(dst->layout == src->layout);
      assert(dst->dims[0] == src->dims[0]);     // C
      assert(dst->dims[1] == src->dims[1] * 2); // H
      assert(dst->dims[2] == src->dims[2] * 2); // W
    }

    std::shared_ptr<Tensor> getDst() const override { return dst; }
  };

  class CPUUpsampleNode : public UpsampleNode
  {
  public:
    CPUUpsampleNode(const Ref<Device>& device,
                    const std::string& name,
                    const std::shared_ptr<Tensor>& src,
                    const std::shared_ptr<Tensor>& dst);

    void execute() override;
  };

} // namespace oidn
