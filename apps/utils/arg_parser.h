// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace oidn {

  // Command-line argument parser
  class ArgParser
  {
  private:
    int argc;
    char** argv;
    int pos;

  public:
    ArgParser(int argc, char* argv[])
      : argc(argc), argv(argv),
        pos(1)
    {}

    bool hasNext() const
    {
      return pos < argc;
    }

    std::string getNext()
    {
      if (pos < argc)
        return argv[pos++];
      else
        throw std::invalid_argument("argument expected");
    }

    std::string getNextOpt()
    {
      std::string str = getNext();
      if (str.empty() || str[0] != '-')
        throw std::invalid_argument("option expected");
      return str.substr(str.find_first_not_of("-"));
    }

    std::string getNextValue()
    {
      std::string str = getNext();
      if (!str.empty() && str[0] == '-')
        throw std::invalid_argument("value expected");
      return str;
    }

    int getNextValueInt()
    {
      std::string str = getNextValue();
      return atoi(str.c_str());
    }

    float getNextValueFloat()
    {
      std::string str = getNextValue();
      return atof(str.c_str());
    }
  };

} // namespace oidn

