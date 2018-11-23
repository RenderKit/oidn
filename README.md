Open Image Denoise
==================

Version 0.2.0 CLOSED ALPHA - DO *NOT* REDISTRIBUTE\
Intel Corporation


Introduction
============

Open Image Denoise is an open source library for denoising Monte Carlo ray traced images.


Requirements
============

Open Image Denoise requires a CPU with at least SSE 4.2 support.

The software dependencies are:
* Operating system:
  * Linux
  * Windows 7 or later
* C++ compiler with C++11 standard support:
  * GNU Compiler Collection
  * Clang
  * Intel C/C++ Compiler
  * Microsoft Visual C++
* CMake 3.0 or later
* Python 2.7 or later
* Intel Threading Building Blocks (TBB) 2017 or later


Known Issues
============

* Crashes when running on Windows and a CPU with AVX-512 support.
* HDR support is currently only a placeholder. It is functional, but the
  quality is low. This will be addressed in a future version.
* Inputs with only color and albedo are not supported yet.
* Windows versions earlier than 7 are not supported yet. This restriction will
  be lifted in a future release.
* macOS is not supported yet. Support will be added in a future release.


Examples
========

C API
-----

```cpp
#include <OpenImageDenoise/oidn.h>

...

// Create an Open Image Denoise device
OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);

// Create an AI denoising filter
OIDNFilter filter = oidnNewFilter(device, "Autoencoder");
oidnSetSharedFilterImage(filter, "color",  colorPtr,  OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
oidnSetSharedFilterImage(filter, "albedo", albedoPtr, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // optional
oidnSetSharedFilterImage(filter, "normal", normalPtr, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // optional
oidnSetSharedFilterImage(filter, "output", outputPtr, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
oidnSetFilter1i(filter, "hdr", 1); // for HDR inputs
oidnCommitFilter(filter);

// Filter the image
oidnExecuteFilter(filter);

// Check for errors
const char* errorMessage;
if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
  printf("Error: %s\n", errorMessage);

// Cleanup
oidnReleaseFilter(filter);
oidnReleaseDevice(device);
```

C++ API
-------

```cpp
#include <OpenImageDenoise/oidn.hpp>

...

// Create an Open Image Denoise device
oidn::DeviceRef device = oidn::newDevice(oidn::DeviceType::CPU);

// Create an AI denoising filter
oidn::FilterRef filter = device.newFilter("Autoencoder");
filter.setImage("color",  colorPtr,  oidn::Format::Float3, width, height);
filter.setImage("albedo", albedoPtr, oidn::Format::Float3, width, height); // optional
filter.setImage("normal", normalPtr, oidn::Format::Float3, width, height); // optional
filter.setImage("output", outputPtr, oidn::Format::Float3, width, height);
filter.set1i("hdr", 1); // for HDR inputs
filter.commit();

// Filter the image
filter.execute();

// Check for errors
const char* errorMessage;
if (device.getError(&errorMessage) != oidn::Error::None)
  std::cout << "Error: " << errorMessage << std::endl;
```