// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include <exception>

OIDN_NAMESPACE_BEGIN

  class Exception : public std::exception
  {
  public:
    Exception(Error error, const char* message)
      : error(error),
        message(std::make_shared<std::string>(message)) {}

    Exception(Error error, const std::string& message)
      : error(error),
        message(std::make_shared<std::string>(message)) {}

    Error code() const noexcept
    {
      return error;
    }

    const char* what() const noexcept override;

  private:
    Error error;

    // Exceptions must have noexcept copy constructors, so we cannot use std::string directly
    std::shared_ptr<std::string> message;
  };

OIDN_NAMESPACE_END
