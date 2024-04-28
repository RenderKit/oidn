// Copyright 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rtlightmap_filter.h"

#if defined(OIDN_FILTER_RTLIGHTMAP)
  #include "weights/rtlightmap_hdr.h"
  #include "weights/rtlightmap_dir.h"
#endif

OIDN_NAMESPACE_BEGIN

  RTLightmapFilter::RTLightmapFilter(const Ref<Device>& device)
    : UNetFilter(device)
  {
    hdr = true;

  #if defined(OIDN_FILTER_RTLIGHTMAP)
    models.hdr = {blobs::weights::rtlightmap_hdr};
    models.dir = {blobs::weights::rtlightmap_dir};
  #endif
  }

  std::shared_ptr<TransferFunction> RTLightmapFilter::newTransferFunc()
  {
    if (hdr)
      return std::make_shared<TransferFunction>(TransferFunction::Type::Log);
    else
      return std::make_shared<TransferFunction>(TransferFunction::Type::Linear);
  }

  void RTLightmapFilter::setImage(const std::string& name, const Ref<Image>& image)
  {
    if (name == "color")
      setParam(color, image);
    else if (name == "output")
      setParam(output, image);
    else
      device->printWarning("unknown filter parameter or type mismatch: '" + name + "'");

    dirty = true;
  }

  void RTLightmapFilter::unsetImage(const std::string& name)
  {
    if (name == "color")
      removeParam(color);
    else if (name == "output")
      removeParam(output);
    else
      device->printWarning("unknown filter parameter or type mismatch: '" + name + "'");

    dirty = true;
  }

  void RTLightmapFilter::setInt(const std::string& name, int value)
  {
    if (name == "directional")
    {
      setParam(directional, value);
      hdr = !directional;
    }
    else
      UNetFilter::setInt(name, value);

    dirty = true;
  }

  int RTLightmapFilter::getInt(const std::string& name)
  {
    if (name == "directional")
      return directional;
    else
      return UNetFilter::getInt(name);
  }

OIDN_NAMESPACE_END
