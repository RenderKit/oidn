// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include <unordered_set>

#define OIDN_DECLARE_INIT_MODULE(name) \
  OIDN_API_EXPORT void OIDN_CONCAT(oidn_init_module_##name##_v, OIDN_VERSION)()

namespace oidn {

  class ModuleLoader
  {
  public:
    ModuleLoader();

    bool load(const std::string& name);
    
  private:
    static void* getSymbolAddress(void* module, const std::string& name);
    static void closeModule(void* module);

    // Returns the absolute path of the module that contains the given address
    // If address is nullptr, returns the path of this module
    static std::string getModulePath(void* address = nullptr);

    std::string modulePathPrefix; // absolute path of the module directory with trailing path separator
    std::unordered_set<std::string> modules; // loaded module names
  };

} // namespace oidn