// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "filter.h"

namespace oidn {

  void Filter::setProgressMonitorFunction(ProgressMonitorFunction func, void* userPtr)
  {
    progressFunc = func;
    progressUserPtr = userPtr;
  }

  void Filter::setParam(int& dst, int src)
  {
    dirtyParam |= dst != src;
    dst = src;
  }

  void Filter::setParam(bool& dst, int src)
  {
    dirtyParam |= dst != bool(src);
    dst = src;
  }

  void Filter::setParam(Image& dst, const Image& src)
  {
    // The image parameter is *not* dirty if only the pointer changes (except to/from nullptr)
    dirtyParam |= (!dst && src) || (dst && !src) ||
                  (dst.width != src.width) || (dst.height != src.height) ||
                  (dst.format != src.format);
    dst = src;
  }

  void Filter::removeParam(Image& dst)
  {
    dirtyParam |= dst;
    dst = Image();
  }

  void Filter::setParam(Data& dst, const Data& src)
  {
    dirtyParam = dst || src;
    dst = src;
  }

  void Filter::removeParam(Data& dst)
  {
    dirtyParam |= dst;
    dst = Data();
  }

} // namespace oidn
