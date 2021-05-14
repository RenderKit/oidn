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

  void Filter::setParam(std::shared_ptr<Image>& dst, const std::shared_ptr<Image>& src)
  {
    // The image parameter is *not* dirty if only the pointer changes (except to/from nullptr)
    dirtyParam |= (!dst && src && *src) || (dst && (!src || !(*src))) ||
                  (dst && src && *src &&
                   ((dst->width != src->width) || (dst->height != src->height) ||
                    (dst->format != src->format)));

    if (src && *src)
      dst = src;
    else
      dst = nullptr;
  }

  void Filter::removeParam(std::shared_ptr<Image>& dst)
  {
    dirtyParam |= bool(dst);
    dst = nullptr;
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
