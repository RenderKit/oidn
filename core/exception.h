// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include <exception>

OIDN_NAMESPACE_BEGIN

  class Exception : public std::exception
  {
  public:
    Exception(Error error, const char* message)
      : error(error), message(message) {}

    Error code() const noexcept
    {
      return error;
    }

    const char* what() const noexcept override;

  private:
    Error error;
    const char* message;
  };

OIDN_NAMESPACE_END
