Compilation
===========

The latest Intel Open Image Denoise sources are always available at the
[Intel Open Image Denoise GitHub repository](http://github.com/OpenImageDenoise/oidn).
The default `master` branch should always point to the latest tested bugfix
release.


Prerequisites
-------------

You can clone the latest Intel Open Image Denoise sources using Git with the
[Git Large File Storage (LFS)](https://git-lfs.github.com/) extension installed:

    git clone --recursive https://github.com/OpenImageDenoise/oidn.git

Please note that installing the Git LFS extension is *required* to correctly
clone the repository. Cloning without Git LFS will seemingly succeed but
actually some of the files will be invalid and thus compilation will fail.

Intel Open Image Denoise currently supports 64-bit Linux, Windows, and macOS
operating systems. Before you can build Intel Open Image Denoise you need the
following basic prerequisites:

-   [CMake](http://www.cmake.org) 3.15 or newer

-   A C++11 compiler (we recommend using a Clang-based compiler but also support
    GCC and Microsoft Visual Studio 2015 and newer)

-   Python 3

To build support for different types of CPUs and GPUs, the following additional
prerequisites are needed:

#### CPU device: {-}

-   [Intel® SPMD Program Compiler (ISPC)](http://ispc.github.io) 1.14.1 or
    newer. Please obtain a release of ISPC from the
    [ISPC downloads page](https://ispc.github.io/downloads.html). The build
    system looks for ISPC in the `PATH` and in the directory right "next to" the
    checked-out Intel Open Image Denoise sources. For example, if Intel Open
    Image Denoise is in `~/Projects/oidn`, ISPC will also be searched in
    `~/Projects/ispc-v1.14.1-linux`. Alternatively set the CMake variable
    `ISPC_EXECUTABLE` to the location of the ISPC compiler.

-   [Intel® Threading Building Blocks](https://github.com/oneapi-src/oneTBB)
    (TBB) 2017 or newer

#### SYCL device for Intel GPUs: {-}

-   [Intel® oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
    2024.0 or newer, or the open source
    [oneAPI DPC++ Compiler 2023-09-22](https://github.com/intel/llvm/releases/tag/nightly-2023-09-22).
    Other SYCL compilers are *not* supported. The open source version of the
    compiler is more up-to-date but less stable, so we *strongly* recommend to
    use the exact version listed here, and on Linux we also recommend to
    rebuild it from source with the `--disable-fusion` flag.

-   Intel® Graphics Offline Compiler for OpenCL™ Code (OCLOC)
    -   Windows:
        Version [31.0.101.5082](https://registrationcenter-download.intel.com/akdlm/IRC_NAS/77a13ae6-6100-4ddc-b069-0086ff44730c/ocloc_win_101.5082.zip)
        or newer as a
        [standalone component of Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html),
        which must be extracted and its contents added to the `PATH`.
        Also included with
        [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit).

    -   Linux: Included with [Intel® software for General Purpose GPU capabilities](https://dgpu-docs.intel.com)
        release [20230918](https://dgpu-docs.intel.com/releases/stable_704_30_20230918.html) or newer
        (install at least `intel-opencl-icd` on Ubuntu, `intel-ocloc` on RHEL or SLES).
        Also available with
        [Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver](https://github.com/intel/compute-runtime).

-   If using Intel® oneAPI DPC++/C++ Compiler:
    [CMake](http://www.cmake.org) 3.25.2 or newer

-   [Ninja](https://ninja-build.org) or Make as the CMake generator. The Visual
    Studio generator is *not* supported.

#### CUDA device for NVIDIA GPUs: {-}

-   [CMake](http://www.cmake.org) 3.18 or newer

-   [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 11.8 or newer

#### HIP device for AMD GPUs: {-}

-   [CMake](http://www.cmake.org) 3.21 or newer

-   [Ninja](https://ninja-build.org) or Make as the CMake generator. The Visual
    Studio generator is *not* supported.

-   [AMD ROCm (HIP SDK)](https://rocm.docs.amd.com) v5.5.0 or newer.

#### Metal device for Apple GPUs: {-}

-   [CMake](http://www.cmake.org) 3.21 or newer

-   [Xcode](https://developer.apple.com/xcode/) 14 or newer

Depending on your operating system, you can install some required dependencies
(e.g., TBB) using `yum` or `apt-get` on Linux, [Homebrew](https://brew.sh) or
[MacPorts](https://www.macports.org) on macOS, and [`vcpkg`](https://vcpkg.io)
on Windows. For the other dependencies please download the necessary packages
or installers and follow the included instructions.


Compiling on Linux/macOS
------------------------

If you are building with SYCL support on Linux, make sure that the DPC++
compiler is properly set up. The open source oneAPI DPC++ Compiler can be
downloaded and simply extracted. However, before using the compiler, the
environment must be set up as well with the following command:

    source ./dpcpp_compiler/startup.sh

The `startup.sh` script will put `clang` and `clang++` from the
oneAPI DPC++ Compiler into your `PATH`.

Alternatively, if you have installed Intel® oneAPI DPC++/C++ Compiler instead,
you can set up the compiler by sourcing the `vars.sh` script in the `env`
directory of the compiler install directory, for example,

    source /opt/intel/oneAPI/compiler/latest/env/vars.sh

This script will put the `icx` and `icpx` compiler executables from the
Intel(R) oneAPI DPC++/C++ Compiler in your `PATH`.

-   Create a build directory, and go into it using a command prompt

        mkdir oidn/build
        cd oidn/build

    (We do recommend having separate build directories for different
    configurations such as release, debug, etc.).

-   CMake will use the default compiler, which on most Linux machines is `gcc`,
    but it can be switched to `clang` by executing the following:

        cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..

    If you are building with SYCL support, you must set the DPC++ compiler
    (`clang`/`clang++` or `icx`/`icpx`) as the C/C++ compiler here. Note that
    the compiler variables cannot be changed after the first `cmake` or `ccmake`
    run.

-   Open the CMake configuration dialog

        ccmake ..

-   Make sure to properly set the build mode and enable the components and
    options you need. By default only CPU support is built, so SYCL and other
    device support must be enabled manually (e.g. with the `OIDN_DEVICE_SYCL`
    option). Then type 'c'onfigure and 'g'enerate. When back on the command
    prompt, build the library using

        ninja


### Entitlements on macOS

macOS requires notarization of applications as a security mechanism, and
[entitlements must be declared](https://developer.apple.com/documentation/bundleresources/entitlements)
during the notarization process.
Intel Open Image Denoise uses just-in-time compilation through [oneDNN](https://github.com/oneapi-src/oneDNN) and requires the following entitlements:

-    [`com.apple.security.cs.allow-jit`](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_security_cs_allow-jit)
-    [`com.apple.security.cs.allow-unsigned-executable-memory`](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_security_cs_allow-unsigned-executable-memory)
-    [`com.apple.security.cs.disable-executable-page-protection`](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_security_cs_disable-executable-page-protection)


Compiling on Windows
--------------------

If you are building with SYCL support, make sure that the DPC++ compiler is
properly set up. The open source oneAPI DPC++ Compiler can be downloaded and
simply extracted. However, before using the compiler, the environment must be
set up. To achieve this, open the "x64 Native Tools Command Prompt for VS"
that ships with Visual Studio and execute the following commands:

    set "DPCPP_DIR=path_to_dpcpp_compiler"
    set "PATH=%DPCPP_DIR%\bin;%PATH%"
    set "PATH=%DPCPP_DIR%\lib;%PATH%"
    set "CPATH=%DPCPP_DIR%\include;%CPATH%"
    set "INCLUDE=%DPCPP_DIR%\include;%INCLUDE%"
    set "LIB=%DPCPP_DIR%\lib;%LIB%"

The `path_to_dpcpp_compiler` should point to the unpacked oneAPI DPC++
Compiler.

Alternatively, if you have installed Intel® oneAPI DPC++/C++ Compiler instead,
you can either open a regular "Command Prompt" and execute the `vars.bat` script
in the `env` directory of the compiler install directory, for example

    C:\Program Files (x86)\Intel\oneAPI\compiler\latest\env\vars.bat

or simply open the installed "Intel oneAPI command prompt for Intel 64 for Visual Studio".
Either way, the `icx` compiler executable from the Intel® oneAPI DPC++/C++ Compiler
will be added to your `PATH`.

On Windows we highly recommend to use Ninja as the CMake generator because not
all devices can be built using the Visual Studio generator (e.g. SYCL).

-   Create a build directory, and go into it using a Visual Studio command prompt

        mkdir oidn/build
        cd oidn/build

    (We do recommend having separate build directories for different
    configurations such as release, debug, etc.).

-   CMake will use the default compiler, which on most Windows machines is
    MSVC, but it can be switched to `clang` by executing the following:

        cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..

    If you are building with SYCL support, you must set the DPC++ compiler
    (`clang`/`clang++` or `icx`) as the C/C++ compiler here. Note that
    the compiler variables cannot be changed after the first `cmake` or
    `cmake-gui` run.

-   Open the CMake GUI (`cmake-gui.exe`)

        cmake-gui ..

-   Make sure to properly set the build mode and enable the components and
    options you need. By default only CPU support is built, so SYCL and other
    device support must be enabled manually (e.g. `OIDN_DEVICE_SYCL` option).
    Then click on Configure and Generate. When back on the command prompt, build
    the library using

        ninja


CMake Configuration
-------------------

The following list describes the options that can be configured in CMake:

- `CMAKE_BUILD_TYPE`: Can be used to switch between Debug mode
  (Debug), Release mode (Release) (default), and Release mode with
  enabled assertions and debug symbols (RelWithDebInfo).

- `OIDN_STATIC_LIB`: Build Open Image Denoise as a static (if only CPU support
  is enabled) or a hybrid static/shared (if GPU support is enabled as well)
  library.

- `OIDN_API_NAMESPACE`: Specifies a namespace to put all Intel Open Image
  Denoise API symbols inside. This is also added as an outer namespace for the
  C++ wrapper API. By default no namespace is used and plain C symbols are
  exported.

- `OIDN_DEVICE_CPU`: Enable CPU device support (ON by default).

- `OIDN_DEVICE_SYCL`: Enable SYCL device support for Intel GPUs (OFF by
  default).

- `OIDN_DEVICE_SYCL_AOT`: Enable ahead-of-time (AOT) compilation for SYCL
  kernels (ON by default). Turning this off removes dependency on OCLOC at
  build time and decreases binary size but significantly increases
  initialization time at runtime, so it is recommended only for development.

- `OIDN_DEVICE_CUDA`: Enable CUDA device support for NVIDIA GPUs (OFF by
  default).

- `OIDN_DEVICE_CUDA_API`: Use the CUDA driver API (`Driver`, default), the
  static CUDA runtime library (`RuntimeStatic`), or the shared CUDA runtime
  library (`RuntimeShared`).

- `OIDN_DEVICE_HIP`: Enable HIP device support for AMD GPUs (OFF by
  default).

- `OIDN_DEVICE_METAL`: Enable Metal device support for Apple GPUs (OFF by
  default).

- `OIDN_FILTER_RT`: Include the trained weights of the `RT` filter in the build
  (ON by default). Turning this OFF significantly decreases the size of the
  library binary, while the filter remains functional if the weights are set by
  the user at runtime.

- `OIDN_FILTER_RTLIGHTMAP`: Include the trained weights of the `RTLightmap`
  filter in the build (ON by default).

- `OIDN_APPS`: Enable building example and test applications (ON by default).

- `OIDN_APPS_OPENIMAGEIO`: Enable [OpenImageIO](http://openimageio.org/)
  support in the example and test applications to be able to load/save
  OpenEXR, PNG, and other image file formats (OFF by default).

- `OIDN_INSTALL_DEPENDENCIES`: Enable installing the dependencies (e.g. TBB,
  SYCL runtime) as well.

- `TBB_ROOT`: The path to the TBB installation (autodetected by default).

- `ROCM_PATH`: The path to the ROCm installation (autodetected by default).

- `OPENIMAGEIO_ROOT`: The path to the OpenImageIO installation (autodetected by
  default).
