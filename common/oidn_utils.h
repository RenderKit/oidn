// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "include/OpenImageDenoise/oidn.hpp"
#include <iostream>

namespace oidn {

#if defined(OIDN_API_NAMESPACE)
  // Introduce all names from the custom API namespace
  OIDN_NAMESPACE_USING
#endif

  // Returns the size of a format in bytes
  size_t getFormatSize(Format format);

  std::ostream& operator <<(std::ostream& sm, Format format);

  std::ostream& operator <<(std::ostream& sm, DeviceType deviceType);
  std::istream& operator >>(std::istream& sm, DeviceType& deviceType);

} // namespace oidn