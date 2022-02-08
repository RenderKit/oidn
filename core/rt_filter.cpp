// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rt_filter.h"

// Default weights
#if defined(OIDN_FILTER_RT)
  #include "weights/rt_hdr.h"
  #include "weights/rt_hdr_alb.h"
  #include "weights/rt_hdr_alb_nrm.h"
  #include "weights/rt_hdr_calb_cnrm.h"
  #include "weights/rt_ldr.h"
  #include "weights/rt_ldr_alb.h"
  #include "weights/rt_ldr_alb_nrm.h"
  #include "weights/rt_ldr_calb_cnrm.h"
  #include "weights/rt_alb.h"
  #include "weights/rt_nrm.h"
#endif

namespace oidn {

  RTFilter::RTFilter(const Ref<Device>& device)
    : UNetFilter(device)
  {
  #if defined(OIDN_FILTER_RT)
    defaultWeights.hdr           = blobs::weights::rt_hdr;
    defaultWeights.hdr_alb       = blobs::weights::rt_hdr_alb;
    defaultWeights.hdr_alb_nrm   = blobs::weights::rt_hdr_alb_nrm;
    defaultWeights.hdr_calb_cnrm = blobs::weights::rt_hdr_calb_cnrm;
    defaultWeights.ldr           = blobs::weights::rt_ldr;
    defaultWeights.ldr_alb       = blobs::weights::rt_ldr_alb;
    defaultWeights.ldr_alb_nrm   = blobs::weights::rt_ldr_alb_nrm;
    defaultWeights.ldr_calb_cnrm = blobs::weights::rt_ldr_calb_cnrm;
    defaultWeights.alb           = blobs::weights::rt_alb;
    defaultWeights.nrm           = blobs::weights::rt_nrm;
  #endif
  }

  std::shared_ptr<TransferFunction> RTFilter::getTransferFunc()
  {
    if (srgb || (!color && normal))
      return std::make_shared<TransferFunction>(TransferFunction::Type::Linear);
    else if (hdr)
      return std::make_shared<TransferFunction>(TransferFunction::Type::PU);
    else
      return std::make_shared<TransferFunction>(TransferFunction::Type::SRGB);
  }

  void RTFilter::setImage(const std::string& name, const std::shared_ptr<Image>& image)
  {
    if (name == "color")
      setParam(color, image);
    else if (name == "albedo")
      setParam(albedo, image);
    else if (name == "normal")
      setParam(normal, image);
    else if (name == "output")
      setParam(output, image);
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void RTFilter::removeImage(const std::string& name)
  {
    if (name == "color")
      removeParam(color);
    else if (name == "albedo")
      removeParam(albedo);
    else if (name == "normal")
      removeParam(normal);
    else if (name == "output")
      removeParam(output);
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void RTFilter::set1i(const std::string& name, int value)
  {
    if (name == "hdr")
      setParam(hdr, value);
    else if (name == "srgb")
      setParam(srgb, value);
    else if (name == "cleanAux")
      setParam(cleanAux, value);
    else if (name == "maxMemoryMB")
      setParam(maxMemoryMB, value);
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  int RTFilter::get1i(const std::string& name)
  {
    if (name == "hdr")
      return hdr;
    else if (name == "srgb")
      return srgb;
    else if (name == "cleanAux")
      return cleanAux;
    else if (name == "maxMemoryMB")
      return maxMemoryMB;
    else if (name == "alignment")
      return alignment;
    else if (name == "overlap")
      return overlap;
    else
      throw Exception(Error::InvalidArgument, "unknown filter parameter");
  }

} // namespace oidn
