// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

OIDN_NAMESPACE_BEGIN

  // Base class for verbose classes
  class Verbose
  {
  public:
    Verbose(int v = 0) : verbose(v) {}

    void setVerbose(int v) { verbose = v; }
    bool isVerbose(int v = 1) const { return v <= verbose; }

    void print(const std::string& message)
    {
      if (isVerbose())
        std::cout << message << std::endl;
    }

    void printWarning(const std::string& message)
    {
      if (isVerbose())
        std::cerr << "Warning: " << message << std::endl;
    }

    void printError(const std::string& message)
    {
      if (isVerbose())
        std::cerr << "Error: " << message << std::endl;
    }

    void printDebug(const std::string& message)
    {
      if (isVerbose(2))
        std::cout << message << std::endl;
    }

  protected:
    int verbose;
  };

OIDN_NAMESPACE_END