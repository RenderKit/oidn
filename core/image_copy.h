// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "image.h"

OIDN_NAMESPACE_BEGIN

  class ImageCopy : public Op
  {
  public:
    void setSrc(const std::shared_ptr<Image>& src) { this->src = src; }
    void setDst(const std::shared_ptr<Image>& dst) { this->dst = dst; }

  protected:
    void check()
    {
      if (!src || !dst)
        throw std::logic_error("image copy source/destination not set");
      if (dst->getH() < src->getH() || dst->getW() < src->getW())
        throw std::out_of_range("image copy destination smaller than the source");
    }

    std::shared_ptr<Image> src;
    std::shared_ptr<Image> dst;
  };

OIDN_NAMESPACE_END
