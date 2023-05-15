// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include "arg_parser.h"

OIDN_NAMESPACE_BEGIN

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
    size_t pos = str.find_first_not_of("-");
    if (pos == 0 || pos == std::string::npos)
      throw std::invalid_argument("option expected");
    return str.substr(pos);
  }

  template<>
  std::string ArgParser::getNextValue()
  {
    std::string str = getNext();
    if (!str.empty() && str[0] == '-')
      throw std::invalid_argument("value expected");
    return str;
  }

OIDN_NAMESPACE_END

