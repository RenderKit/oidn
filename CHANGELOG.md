Version History
---------------

### Changes in v1.4.1:

-   Fixed crash when in-place denoising images with certain unusual resolutions
-   Fixed compile error when building for Apple Silicon using some unofficial
    builds of ISPC

### Changes in v1.4.0:

-   Improved fine detail preservation
-   Added the `cleanAux` filter parameter for further improving quality when the
    auxiliary feature (albedo, normal) images are noise-free
-   Added support for denoising auxiliary feature images, which can be used 
    together with the new `cleanAux` parameter for improving quality when the
    auxiliary images are noisy (recommended for final frame denoising)
-   Normals are expected to be in the [-1, 1] range (but still do not have to
    be normalized)
-   Added the `oidnUpdateFilterData` function which must be called when the
    contents of an opaque data parameter bound to a filter (e.g. `weights`) has
    been changed after committing the filter
-   Added the `oidnRemoveFilterImage` and `oidnRemoveFilterData` functions for
    removing previously set image and opaque data parameters of filters
-   Reduced the overhead of `oidnCommitFilter` to zero in some cases (e.g. when
    changing already set image buffers/pointers or the `inputScale` parameter)
-   Reduced filter memory consumption by about 35%
-   Reduced total memory consumption significantly when using multiple filters
    that belong to the same device
-   Reduced the default maximum memory consumption to 3000 MB
-   Added the `OIDN_FILTER_RT` and `OIDN_FILTER_RTLIGHTMAP` CMake options for
    excluding the trained filter weights from the build to significantly
    decrease its size
-   Fixed detection of static TBB builds on Windows
-   Fixed compile error when using future glibc versions
-   Added `oidnBenchmark` option for setting custom resolutions
-   Upgraded to oneTBB 2021.2.0 in the official binaries

### Changes in v1.3.0:

-   Improved denoising quality
    -   Improved sharpness of fine details / less blurriness
    -   Fewer noisy artifacts
-   Slightly improved performance and lowered memory consumption
-   Added directional (e.g. spherical harmonics) lightmap denoising to the
    `RTLightmap` filter
-   Added `inputScale` filter parameter which generalizes the existing
    (and thus now deprecated) `hdrScale` parameter for non-HDR images
-   Added native support for Apple Silicon and the BNNS library on macOS
    (currently requires rebuilding from source)
-   Added `OIDN_NEURAL_RUNTIME` CMake option for setting the neural network
    runtime library
-   Reduced the size of the library binary
-   Fixed compile error on some older macOS versions
-   Upgraded release builds to use oneTBB 2021.1.1
-   Removed tbbmalloc dependency
-   Appended the library version to the name of the directory containing the
    installed CMake files
-   Training:
    -   Faster training performance
    -   Added mixed precision training (enabled by default)
    -   Added efficient data-parallel training on multiple GPUs
    -   Enabled preprocessing datasets multiple times with possibly different
        options
    -   Minor bugfixes

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
