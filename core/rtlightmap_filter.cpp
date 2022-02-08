// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rtlightmap_filter.h"

#if defined(OIDN_FILTER_RTLIGHTMAP)
  #include "weights/rtlightmap_hdr.h"
  #include "weights/rtlightmap_dir.h"
#endif

namespace oidn {

  RTLightmapFilter::RTLightmapFilter(const Ref<Device>& device)
    : UNetFilter(device)
  {
    hdr = true;

  #if defined(OIDN_FILTER_RTLIGHTMAP)
    defaultWeights.hdr = blobs::weights::rtlightmap_hdr;
    defaultWeights.dir = blobs::weights::rtlightmap_dir;
  #endif
  }

  std::shared_ptr<TransferFunction> RTLightmapFilter::getTransferFunc()
  {
    if (hdr)
      return std::make_shared<TransferFunction>(TransferFunction::Type::Log);
    else
      return std::make_shared<TransferFunction>(TransferFunction::Type::Linear);
  }

  void RTLightmapFilter::setImage(const std::string& name, const std::shared_ptr<Image>& image)
  {
    if (name == "color")
      setParam(color, image);
    else if (name == "output")
      setParam(output, image);
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void RTLightmapFilter::removeImage(const std::string& name)
  {
    if (name == "color")
      removeParam(color);
    else if (name == "output")
      removeParam(output);
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void RTLightmapFilter::set1i(const std::string& name, int value)
  {
    if (name == "directional")
    {
      setParam(directional, value);
      hdr = !directional;
    }
    else if (name == "maxMemoryMB")
      setParam(maxMemoryMB, value);
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  int RTLightmapFilter::get1i(const std::string& name)
  {
    if (name == "directional")
      return directional;
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
