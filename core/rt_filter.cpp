// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rt_filter.h"

// Default weights
#if defined(OIDN_FILTER_RT)
  #include "weights/rt_hdr.h"
  #include "weights/rt_hdr_alb.h"
  #include "weights/rt_hdr_alb_nrm.h"
  #include "weights/rt_hdr_calb_cnrm.h"
  #include "weights/rt_hdr_calb_cnrm_hq.h"
  #include "weights/rt_ldr.h"
  #include "weights/rt_ldr_alb.h"
  #include "weights/rt_ldr_alb_nrm.h"
  #include "weights/rt_ldr_calb_cnrm.h"
  #include "weights/rt_alb.h"
  #include "weights/rt_alb_hq.h"
  #include "weights/rt_nrm.h"
  #include "weights/rt_nrm_hq.h"
#endif

OIDN_NAMESPACE_BEGIN

  RTFilter::RTFilter(const Ref<Device>& device)
    : UNetFilter(device)
  {
  #if defined(OIDN_FILTER_RT)
    weightsBlobs.hdr              = blobs::weights::rt_hdr;
    weightsBlobs.hdr_alb          = blobs::weights::rt_hdr_alb;
    weightsBlobs.hdr_alb_nrm      = blobs::weights::rt_hdr_alb_nrm;
    weightsBlobs.hdr_calb_cnrm    = blobs::weights::rt_hdr_calb_cnrm;
    weightsBlobs.hdr_calb_cnrm_hq = blobs::weights::rt_hdr_calb_cnrm_hq;
    weightsBlobs.ldr              = blobs::weights::rt_ldr;
    weightsBlobs.ldr_alb          = blobs::weights::rt_ldr_alb;
    weightsBlobs.ldr_alb_nrm      = blobs::weights::rt_ldr_alb_nrm;
    weightsBlobs.ldr_calb_cnrm    = blobs::weights::rt_ldr_calb_cnrm;
    weightsBlobs.alb              = blobs::weights::rt_alb;
    weightsBlobs.alb_hq           = blobs::weights::rt_alb_hq;
    weightsBlobs.nrm              = blobs::weights::rt_nrm;
    weightsBlobs.nrm_hq           = blobs::weights::rt_nrm_hq;
  #endif
  }

  std::shared_ptr<TransferFunction> RTFilter::newTransferFunc()
  {
    if (srgb || (!color && normal))
      return std::make_shared<TransferFunction>(TransferFunction::Type::Linear);
    else if (hdr)
      return std::make_shared<TransferFunction>(TransferFunction::Type::PU);
    else
      return std::make_shared<TransferFunction>(TransferFunction::Type::SRGB);
  }

  void RTFilter::setImage(const std::string& name, const Ref<Image>& image)
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
      device->printWarning("unknown filter parameter or type mismatch: '" + name + "'");

    dirty = true;
  }

  void RTFilter::unsetImage(const std::string& name)
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
      device->printWarning("unknown filter parameter or type mismatch: '" + name + "'");

    dirty = true;
  }

  void RTFilter::setInt(const std::string& name, int value)
  {
    if (name == "hdr")
      setParam(hdr, value);
    else if (name == "srgb")
      setParam(srgb, value);
    else if (name == "cleanAux")
      setParam(cleanAux, value);
    else
      UNetFilter::setInt(name, value);

    dirty = true;
  }

  int RTFilter::getInt(const std::string& name)
  {
    if (name == "hdr")
      return hdr;
    else if (name == "srgb")
      return srgb;
    else if (name == "cleanAux")
      return cleanAux;
    else
      return UNetFilter::getInt(name);
  }

OIDN_NAMESPACE_END
