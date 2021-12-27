// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"

namespace oidn {

  struct ReorderTile
  {
    int hSrcBegin;
    int wSrcBegin;
    int hDstBegin;
    int wDstBegin;
    int H;
    int W;

    operator ispc::ReorderTile() const
    {
      ispc::ReorderTile res;
      res.hSrcBegin = hSrcBegin;
      res.wSrcBegin = wSrcBegin;
      res.hDstBegin = hDstBegin;
      res.wDstBegin = wDstBegin;
      res.H = H;
      res.W = W;
      return res;
    }
  };

  struct ReorderDesc
  {
    std::string name;
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;
  };

} // namespace oidn
