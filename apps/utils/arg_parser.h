// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

namespace oidn {

  // Command-line argument parser
  class ArgParser
  {
  public:
    ArgParser(int argc, char* argv[]);

    bool hasNext() const;
    std::string getNext();
    std::string getNextOpt();

    template<typename T = std::string>
    T getNextValue()
    {
      return fromString<T>(getNextValue());
    }

  private:
    int argc;
    char** argv;
    int pos;
  };

  template<>
  std::string ArgParser::getNextValue();

} // namespace oidn

