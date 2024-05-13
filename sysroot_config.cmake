# configure_sysroot.cmake

if(OIDN_METAL_IOS)
  set(SYSROOT_PATH "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk")
  set(CMAKE_OSX_SYSROOT ${SYSROOT_PATH})
else()
  set(SYSROOT_PATH "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
  set(CMAKE_OSX_SYSROOT ${SYSROOT_PATH})
endif()

file(WRITE ${CMAKE_BINARY_DIR}/sysroot_config.cmake "set(CMAKE_OSX_SYSROOT ${SYSROOT_PATH})")
