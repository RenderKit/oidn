// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "include/OpenImageDenoise/oidn.hpp"
#include <iostream>

OIDN_NAMESPACE_BEGIN

  // Returns the size of a format in bytes
  size_t getFormatSize(Format format);

  std::ostream& operator <<(std::ostream& sm, Format format);

  std::ostream& operator <<(std::ostream& sm, DeviceType deviceType);
  std::istream& operator >>(std::istream& sm, DeviceType& deviceType);

OIDN_NAMESPACE_END