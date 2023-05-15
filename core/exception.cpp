// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "exception.h"

OIDN_NAMESPACE_BEGIN

  // We *must* define this function here because Exception must have a key function, which is the
  // first non-pure out-of-line virtual function of a type. Otherwise, the type_info would be
  // emitted as a weak symbol and its address may be different in dynamically loaded modules,
  // which would cause exception handling and dynamic_cast to fail.
  const char* Exception::what() const noexcept
  {
    return message->c_str();
  }

OIDN_NAMESPACE_END