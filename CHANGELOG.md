Version History
---------------

### Changes in v1.2.4:

-   Added OIDN_API_NAMESPACE CMake option that allows to put all API functions
    inside a user-defined namespace
-   Fixed bug when TBB_USE_GLIBCXX_VERSION is defined
-   Fixed compile error when using an old compiler which does not support
    OpenMP SIMD
-   Added compatibility with oneTBB 2021
-   Export only necessary symbols on Linux and macOS

### Changes in v1.2.3:

-   Fixed incorrect detection of AVX-512 on macOS (sometimes causing a crash)
-   Fixed inconsistent performance and costly initialization for AVX-512
-   Fixed JIT'ed AVX-512 kernels not showing up correctly in VTune

### Changes in v1.2.2:

-   Fixed unhandled exception when canceling filter execution from the
    progress monitor callback function

### Changes in v1.2.1:

-   Fixed tiling artifacts when in-place denoising (using one of the input
    images as the output) high-resolution (> 1080p) images
-   Fixed ghosting/color bleeding artifacts in black regions when using
    albedo/normal buffers
-   Fixed error when building as a static library (`OIDN_STATIC_LIB` option)
-   Fixed compile error for ISPC 1.13 and later
-   Fixed minor TBB detection issues
-   Fixed crash on pre-SSE4 CPUs when using some recent compilers (e.g. GCC 10)
-   Link C/C++ runtime library dynamically on Windows too by default
-   Renamed example apps (`oidnDenoise`, `oidnTest`)
-   Added benchmark app (`oidnBenchmark`)
-   Fixed random data augmentation seeding in training
-   Fixed training warning with PyTorch 1.5 and later

### Changes in v1.2.0:

-   Added neural network training code
-   Added support for specifying user-trained models at runtime
-   Slightly improved denoising quality (e.g. less ringing artifacts, less
    blurriness in some cases)
-   Improved denoising speed by about 7-38% (mostly depending on the compiler)
-   Added `OIDN_STATIC_RUNTIME` CMake option (for Windows only)
-   Added support for OpenImageIO to the example apps (disabled by default)
-   Added check for minimum supported TBB version
-   Find debug versions of TBB
-   Added testing

### Changes in v1.1.0:

-   Added `RTLightmap` filter optimized for lightmaps
-   Added `hdrScale` filter parameter for manually specifying the mapping
    of HDR color values to luminance levels

### Changes in v1.0.0:

-   Improved denoising quality
    -   More details preserved
    -   Less artifacts (e.g. noisy spots, color bleeding with albedo/normal)
-   Added `maxMemoryMB` filter parameter for limiting the maximum memory
    consumption regardless of the image resolution, potentially at the cost
    of lower denoising speed. This is internally implemented by denoising the
    image in tiles
-   Significantly reduced memory consumption (but slightly lower performance)
    for high resolutions (> 2K) by default: limited to about 6 GB
-   Added `alignment` and `overlap` filter parameters that can be queried for
    manual tiled denoising
-   Added `verbose` device parameter for setting the verbosity of the console
    output, and disabled all console output by default
-   Fixed crash for zero-sized images

### Changes in v0.9.0:

-   Reduced memory consumption by about 38%
-   Added support for progress monitor callback functions
-   Enabled fully concurrent execution when using multiple devices
-   Clamp LDR input and output colors to 1
-   Fixed issue where some memory allocation errors were not reported

### Changes in v0.8.2:

-   Fixed wrong HDR output when the input contains infinities/NaNs
-   Fixed wrong output when multiple filters were executed concurrently on
    separate devices with AVX-512 support. Currently the filter executions are
    serialized as a temporary workaround, and a full fix will be included in a
    future release.
-   Added `OIDN_STATIC_LIB` CMake option for building as a static library
    (requires CMake 3.13.0 or later)
-   Fixed CMake error when adding the library with add_subdirectory() to a project

### Changes in v0.8.1:

-   Fixed wrong path to TBB in the generated CMake configs
-   Fixed wrong rpath in the binaries
-   Fixed compile error on some macOS systems
-   Fixed minor compile issues with Visual Studio
-   Lowered the CPU requirement to SSE4.1
-   Minor example update

### Changes in v0.8.0:

-   Initial beta release
