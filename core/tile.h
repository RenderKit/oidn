// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"

OIDN_NAMESPACE_BEGIN

  struct Tile
  {
    int hSrcBegin;
    int wSrcBegin;
    int hDstBegin;
    int wDstBegin;
    int H;
    int W;
  };

OIDN_NAMESPACE_END