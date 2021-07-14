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
operating systems. In addition, before you can build Intel Open Image Denoise
you need the following prerequisites:

-   [CMake](http://www.cmake.org) 3.1 or later

-   A C++11 compiler (we recommend using Clang, but also support GCC, Microsoft
    Visual Studio 2015 or later, and
    [Intel® C++ Compiler](https://software.intel.com/en-us/c-compilers) 17.0 or
    later)

-   [Intel® SPMD Program Compiler (ISPC)](http://ispc.github.io), version 1.14.1
    or later. Please obtain a release of ISPC from the [ISPC downloads
    page](https://ispc.github.io/downloads.html). The build system looks for
    ISPC in the `PATH` and in the directory right "next to" the checked-out
    Intel Open Image Denoise sources.^[For example, if Intel Open Image Denoise
    is in `~/Projects/oidn`, ISPC will also be searched in `~/Projects/ispc-v1.14.1-linux`]
    Alternatively set the CMake variable `ISPC_EXECUTABLE` to the location of
    the ISPC compiler.

-   Python 2.7 or later

-   [Intel® Threading Building Blocks](https://www.threadingbuildingblocks.org/)
    (TBB) 2017 or later

Depending on your Linux distribution you can install these dependencies
using `yum` or `apt-get`. Some of these packages might already be installed or
might have slightly different names.

Type the following to install the dependencies using `yum`:

    sudo yum install cmake
    sudo yum install tbb-devel

Type the following to install the dependencies using `apt-get`:

    sudo apt-get install cmake-curses-gui
    sudo apt-get install libtbb-dev

Under macOS these dependencies can be installed using
[MacPorts](http://www.macports.org/):

    sudo port install cmake tbb

Under Windows please directly use the appropriate installers or packages for
[CMake](https://cmake.org/download/),
[Python](https://www.python.org/downloads/),
and [TBB](https://github.com/01org/tbb/releases).


Compiling on Linux/macOS
------------------------

Assuming the above prerequisites are all fulfilled, building Intel Open Image
Denoise through CMake is easy:

-   Create a build directory, and go into it

        mkdir oidn/build
        cd oidn/build

    (We do recommend having separate build directories for different
    configurations such as release, debug, etc.).

-   The compiler CMake will use by default will be whatever the `CC` and
    `CXX` environment variables point to. Should you want to specify a
    different compiler, run cmake manually while specifying the desired
    compiler. The default compiler on most Linux machines is `gcc`, but
    it can be pointed to `clang` instead by executing the following:

        cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang ..

    CMake will now use Clang instead of GCC. If you are OK with using
    the default compiler on your system, then simply skip this step.
    Note that the compiler variables cannot be changed after the first
    `cmake` or `ccmake` run.

-   Open the CMake configuration dialog

        ccmake ..

-   Make sure to properly set the build mode and enable the components you
    need, etc.; then type 'c'onfigure and 'g'enerate. When back on the
    command prompt, build it using

        make

-   You should now have `libOpenImageDenoise.so` on Linux or
    `libOpenImageDenoise.dylib` on macOS, and a set of example applications
    as well.


Entitlements on macOS
---------------------

macOS requires notarization of applications as a security mechanism, and 
[entitlements must be declared](https://developer.apple.com/documentation/bundleresources/entitlements)
during the notarization process.  
Intel Open Image Denoise uses just-in-time compilaton through [oneDNN](https://github.com/oneapi-src/oneDNN) and requires the following entitlements:

-    [`com.apple.security.cs.allow-jit`](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_security_cs_allow-jit)
-    [`com.apple.security.cs.allow-unsigned-executable-memory`](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_security_cs_allow-unsigned-executable-memory)
-    [`com.apple.security.cs.disable-executable-page-protection`](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_security_cs_disable-executable-page-protection)


Compiling on Windows
--------------------

On Windows using the CMake GUI (`cmake-gui.exe`) is the most convenient way to
configure Intel Open Image Denoise and to create the Visual Studio solution
files:

-   Browse to the Intel Open Image Denoise sources and specify a build directory
    (if it does not exist yet CMake will create it).

-   Click "Configure" and select as generator the Visual Studio version you
    have (Intel Open Image Denoise needs Visual Studio 14 2015 or newer), for
    Win64 (32-bit builds are not supported), e.g., "Visual Studio 15 2017 Win64".

-   If the configuration fails because some dependencies could not be found
    then follow the instructions given in the error message, e.g., set the
    variable `TBB_ROOT` to the folder where TBB was installed.

-   Optionally change the default build options, and then click "Generate" to
    create the solution and project files in the build directory.

-   Open the generated `OpenImageDenoise.sln` in Visual Studio, select the
    build configuration and compile the project.


Alternatively, Intel Open Image Denoise can also be built without any GUI,
entirely on the console. In the Visual Studio command prompt type:

    cd path\to\oidn
    mkdir build
    cd build
    cmake -G "Visual Studio 15 2017 Win64" [-D VARIABLE=value] ..
    cmake --build . --config Release

Use `-D` to set variables for CMake, e.g., the path to TBB with "`-D
TBB_ROOT=\path\to\tbb`".


CMake Configuration
-------------------

The default CMake configuration in the configuration dialog should be appropriate
for most usages. The following list describes the options that can be configured
in CMake:

- `CMAKE_BUILD_TYPE`: Can be used to switch between Debug mode
  (Debug), Release mode (Release) (default), and Release mode with
  enabled assertions and debug symbols (RelWithDebInfo).

- `OIDN_STATIC_LIB`: Build Intel Open Image Denoise as a static library (OFF by
  default). When using the statically compiled Intel Open Image Denoise library,
  you either have to use the generated CMake configuration files (recommended),
  or you have to manually define `OIDN_STATIC_LIB` before including the library
  headers in your application.

- `OIDN_STATIC_RUNTIME`: Use the static version of the C/C++ runtime library
  (available only on Windows, OFF by default).

- `OIDN_NEURAL_RUNTIME`: Specifies which neural network runtime library to use: 
  `DNNL` (oneDNN, default) or `BNNS` (available only on macOS).

- `OIDN_API_NAMESPACE`: Specifies a namespace to put all Intel Open Image
  Denoise API symbols inside. By default no namespace is used and plain C
  symbols are exported.

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

- `TBB_ROOT`: The path to the TBB installation (autodetected by default).

- `OPENIMAGEIO_ROOT`: The path to the OpenImageIO installation (autodetected by
  default).
