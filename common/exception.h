// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include "platform.h"

namespace oidn {

  class Exception : public std::exception
  {
  private:
    Error error;
    const char* message;

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
  };

} // namespace oidn
