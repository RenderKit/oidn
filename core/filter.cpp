// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "filter.h"

OIDN_NAMESPACE_BEGIN

  Filter::Filter(const Ref<Device>& device)
    : device(device) {}

  Filter::~Filter()
  {
    // We trim the scratch heaps only here to make filter resolution changes more efficient
    device->trimScratch();
  }

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

  void Filter::setParam(Quality& dst, Quality src)
  {
    dirtyParam |= dst != src;
    dst = src;
  }

  void Filter::setParam(Ref<Image>& dst, const Ref<Image>& src)
  {
    // Check whether the image is accessible by the device
    if (src && *src && !device->isSystemMemorySupported())
    {
      const Storage storage = src->getBuffer() ? src->getBuffer()->getStorage()
                                               : device->getPtrStorage(src->getPtr());
      if (storage == Storage::Undefined)
        throw Exception(Error::InvalidArgument, "image data not accessible by the device, please use OIDNBuffer or device allocator for storage");
    }

    // The image parameter is *not* dirty if only the pointer and/or strides change (except to/from nullptr)
    dirtyParam |= (!dst && src && *src) || (dst && (!src || !(*src))) ||
                  (dst && src && *src &&
                   ((dst->getW() != src->getW()) || (dst->getH() != src->getH()) ||
                    (dst->getFormat() != src->getFormat())));

    if (src && *src)
      dst = src;
    else
      dst = nullptr;
  }

  void Filter::removeParam(Ref<Image>& dst)
  {
    dirtyParam |= bool(dst);
    dst = nullptr;
  }

  void Filter::setParam(Data& dst, const Data& src)
  {
    // Check whether the data is accessible to the host
    if (src && device->getPtrStorage(src.ptr) == Storage::Device)
      throw Exception(Error::InvalidArgument, "the specified data is not accessible to the host, please use host malloc");

    dirtyParam = dst || src;
    dst = src;
  }

  void Filter::removeParam(Data& dst)
  {
    dirtyParam |= dst;
    dst = Data();
  }

OIDN_NAMESPACE_END
