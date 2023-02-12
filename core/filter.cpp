// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "filter.h"

OIDN_NAMESPACE_BEGIN

  Filter::Filter(const Ref<Device>& device) : device(device) {}

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
    // Check whether the image is accessible to the device
    if (src && *src)
    {
      Storage storage = src->getBuffer() ? src->getBuffer()->getStorage() : device->getPointerStorage(src->getData());
      if (storage == Storage::Undefined)
        throw Exception(Error::InvalidArgument, "the specified image is not accessible to the device, please use OIDNBuffer or native device malloc");
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

  void Filter::removeParam(std::shared_ptr<Image>& dst)
  {
    dirtyParam |= bool(dst);
    dst = nullptr;
  }

  void Filter::setParam(Data& dst, const Data& src)
  {
    // Check whether the data is accessible to the host
    if (src && device->getPointerStorage(src.ptr) == Storage::Device)
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
