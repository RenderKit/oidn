Open Image Denoise
==================

Version 0.5.0 CLOSED ALPHA - DO *NOT* REDISTRIBUTE\
Intel Corporation


Introduction
============

Open Image Denoise is an open source library for denoising Monte Carlo ray traced images.


Requirements
============

Open Image Denoise requires a CPU with at least SSE 4.2 support.

The software dependencies are:
- Operating system:
  - Linux
  - Windows 7 or later is recommended
  - macOS
- C++ compiler with C++11 standard support:
  - GNU Compiler Collection
  - Clang
  - Intel C/C++ Compiler
  - Microsoft Visual C++
- CMake 3.0 or later
- Python 2.7 or later
- Intel Threading Building Blocks (TBB) 2017 or later


Known Issues
============

- Inputs with only color and albedo are not supported yet.


Examples
========

C API
-----

```cpp
#include <OpenImageDenoise/oidn.h>

...

// Create an Open Image Denoise device
OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);

// Create a denoising filter for ray traced images
OIDNFilter filter = oidnNewFilter(device, "RT");
oidnSetSharedFilterImage(filter, "color",  colorPtr,  OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
oidnSetSharedFilterImage(filter, "albedo", albedoPtr, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // optional
oidnSetSharedFilterImage(filter, "normal", normalPtr, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // optional
oidnSetSharedFilterImage(filter, "output", outputPtr, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
oidnSetFilter1i(filter, "hdr", 1); // enable HDR mode
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
oidn::DeviceRef device = oidn::newDevice();

// Create a denoising filter for ray traced images
oidn::FilterRef filter = device.newFilter("RT");
filter.setImage("color",  colorPtr,  oidn::Format::Float3, width, height);
filter.setImage("albedo", albedoPtr, oidn::Format::Float3, width, height); // optional
filter.setImage("normal", normalPtr, oidn::Format::Float3, width, height); // optional
filter.setImage("output", outputPtr, oidn::Format::Float3, width, height);
filter.set1i("hdr", 1); // enable HDR mode
filter.commit();

// Filter the image
filter.execute();

// Check for errors
const char* errorMessage;
if (device.getError(&errorMessage) != oidn::Error::None)
  std::cout << "Error: " << errorMessage << std::endl;
```