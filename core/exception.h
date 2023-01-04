// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"
#include <exception>

namespace oidn {

  class Exception : public std::exception
  {
  public:
    Exception(Error error, const char* message)
      : error(error), message(message) {}

    Error code() const noexcept
    {
      return error;
    }

    const char* what() const noexcept override
    {
      return message;
    }

  private:
    Error error;
    const char* message;
  };

} // namespace oidn
