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

#if defined(OIDN_STATIC_LIB)
  #define OIDN_DECLARE_INIT_STATIC_MODULE(name) void init_##name()
#else
  #define OIDN_DECLARE_INIT_STATIC_MODULE(name) OIDN_DECLARE_INIT_MODULE(name)
#endif

OIDN_NAMESPACE_BEGIN

  class ModuleLoader
  {
  public:
    ModuleLoader();

    bool load(const std::string& name);

  private:
  #if defined(_WIN32)
    using Path = std::wstring;
    static constexpr const wchar_t* pathSeps = L"/\\";
  #else
    using Path = std::string;
    static constexpr const char* pathSeps = "/\\";
  #endif

    static void* getSymbolAddress(void* module, const std::string& name);
    static void closeModule(void* module);

    // Returns the absolute path of the module that contains the given address
    // If address is nullptr, returns the path of this module
    static Path getModulePath(void* address = nullptr);

    Path modulePathPrefix; // absolute path of the module directory with trailing path separator
    std::unordered_set<std::string> modules; // loaded module names
  };

OIDN_NAMESPACE_END