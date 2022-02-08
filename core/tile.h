// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"

namespace oidn {

  struct Tile
  {
    int hSrcBegin;
    int wSrcBegin;
    int hDstBegin;
    int wDstBegin;
    int H;
    int W;

    operator ispc::Tile() const
    {
      ispc::Tile res;
      res.hSrcBegin = hSrcBegin;
      res.wSrcBegin = wSrcBegin;
      res.hDstBegin = hDstBegin;
      res.wDstBegin = wDstBegin;
      res.H = H;
      res.W = W;
      return res;
    }
  };

} // namespace oidn