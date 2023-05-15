// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include <unordered_set>

#if defined(_WIN32)
  #define OIDN_MODULE_EXPORT extern "C" __declspec(dllexport)
#else
  #define OIDN_MODULE_EXPORT extern "C" __attribute__ ((visibility ("default")))
#endif

#define OIDN_DECLARE_INIT_MODULE(name) \
  OIDN_MODULE_EXPORT void OIDN_CONCAT(OIDN_NAMESPACE_C, OIDN_CONCAT(_init_module_##name##_v, OIDN_VERSION))()

OIDN_NAMESPACE_BEGIN

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

OIDN_NAMESPACE_END