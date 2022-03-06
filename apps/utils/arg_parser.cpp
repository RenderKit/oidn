// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include "arg_parser.h"

namespace oidn {

  ArgParser::ArgParser(int argc, char* argv[])
    : argc(argc), argv(argv),
      pos(1) {}

  bool ArgParser::hasNext() const
  {
    return pos < argc;
  }

  std::string ArgParser::getNext()
  {
    if (pos < argc)
      return argv[pos++];
    else
      throw std::invalid_argument("argument expected");
  }

  std::string ArgParser::getNextOpt()
  {
    std::string str = getNext();
    if (str.empty() || str[0] != '-')
      throw std::invalid_argument("option expected");
    return str.substr(str.find_first_not_of("-"));
  }

  template<>
  std::string ArgParser::getNextValue()
  {
    std::string str = getNext();
    if (!str.empty() && str[0] == '-')
      throw std::invalid_argument("value expected");
    return str;
  }

} // namespace oidn

