// Copyright 2009-2022 Intel Corporation
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
    weightsBlobs.hdr = blobs::weights::rtlightmap_hdr;
    weightsBlobs.dir = blobs::weights::rtlightmap_dir;
  #endif
  }

  std::shared_ptr<TransferFunction> RTLightmapFilter::newTransferFunc()
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
      device->warning("unknown filter parameter or type mismatch");

    dirty = true;
  }

  void RTLightmapFilter::removeImage(const std::string& name)
  {
    if (name == "color")
      removeParam(color);
    else if (name == "output")
      removeParam(output);
    else
      device->warning("unknown filter parameter or type mismatch");

    dirty = true;
  }

  void RTLightmapFilter::setInt(const std::string& name, int value)
  {
    if (name == "directional")
    {
      setParam(directional, value);
      hdr = !directional;
    }
    else if (name == "maxMemoryMB")
      setParam(maxMemoryMB, value);
    else
      device->warning("unknown filter parameter or type mismatch");

    dirty = true;
  }

  int RTLightmapFilter::getInt(const std::string& name)
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
      throw Exception(Error::InvalidArgument, "unknown filter parameter or type mismatch");
  }

OIDN_NAMESPACE_END
