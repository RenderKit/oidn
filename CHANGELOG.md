Version History
---------------

### Changes in v2.3.3:

-   Added NVIDIA Blackwell GPU support
-   Added AMD RDNA4 GPU support
-   Improved performance for AMD RDNA3 GPUs
-   Added `OIDN_DEPENDENTLOADFLAG` CMake option for setting the
    `DEPENDENTLOADFLAG` linker flag on Windows
-   Added `OIDN_LIBRARY_VERSIONED` CMake option for toggling versioning in the
    Open Image Denoise library files
-   Known issue: performance regression for AMD RDNA2 GPUs

### Changes in v2.3.2:

-   Improved performance for Intel Lunar Lake and Battlemage GPUs
-   Added Intel Panther Lake GPU support
-   Fixed compile error when building with OpenImageIO 3.x

### Changes in v2.3.1:

-   Fixed corrupted output when in-place denoising high-resolution (> 1080p)
    images where the input and output are stored in different shared buffer
    objects (created with `oidnNewSharedBuffer*`) that overlap in memory
-   Fixed issues with cancellation through progress monitor callbacks:
    -   Fixed cancellation requests not being fulfilled on CPU devices since
        `v2.3.0`
    -   Fixed not calling the callback anymore after requesting cancellation,
        while the operation is still being executed
-   Added support for creating shared buffers on Metal devices
-   Enabled accessing system allocated memory for CUDA devices which support this
    feature (see `systemMemorySupported` device parameter)
-   Added LUID support for HIP devices. Importing DX12 and Vulkan buffers is
    now functional when using recent AMD GPU drivers on Windows

### Changes in v2.3.0:

-   Significantly improved image quality of the `RT` filter in *high* quality
    mode for HDR denoising with prefiltering, i.e., the following combinations
    of input features and parameters:
        -   HDR color + albedo + normal + `cleanAux`
        -   albedo
        -   normal
    In these cases a much more complex filter is used, which results in lower
    performance than before (about 2x). To revert to the previous performance
    behavior, please switch to the *balanced* quality mode.
-   Added *fast* quality mode (`OIDN_QUALITY_FAST`) for even higher performance
    (about 1.5-2x) interactive/real-time previews and lower default memory usage
    at the cost of somewhat lower image quality. Currently this is implemented
    for the `RT` filter except prefiltering (albedo, normal). In other cases
    denoising implicitly falls back to *balanced* mode.
-   Added Intel Arrow Lake, Lunar Lake, and Battlemage GPU support
-   Execute `Async` functions asynchronously on CPU devices as well
-   Load/initialize device modules lazily (improves stability)
-   Added `oidnIsCPUDeviceSupported`, `oidnIsSYCLDeviceSupported`,
    `oidnIsCUDADeviceSupported`, `oidnIsHIPDeviceSupported`,
    and `oidnIsMetalDeviceSupported` API functions for checking whether a
    physical device of a particular type is supported
-   Release the CUDA primary context when destroying the device object if using
    the CUDA driver API
-   Added `OIDN_LIBRARY_NAME` CMake option for setting the base name of the Open
    Image Denoise library files
-   Fixed device creation error with `oidnNewDevice` when the default device of
    the specified type (e.g. CUDA) is not supported but there are other
    supported non-default devices of that type in the system
-   Fixed CMake error when building with Metal support using non-Apple Clang
-   Fixed iOS build errors
-   Added support for building with ROCm 6.x
-   `oidnNewCUDADevice` and `oidnNewHIPDevice` no longer accept negative device
    IDs. If the goal is to use the current device, its actual ID needs to be
    passed.
-   Upgraded to oneTBB 2021.12.0 in the official binaries
-   Training:
    -   Improved training performance on CUDA and MPS devices, added `--compile`
        option
    -   Added `--quality` option (`high`, `balanced`, `fast`) for selecting the
        size of the model to train, changed the default from `balanced` to `high`
    -   Added new models to the `--model` option (`unet_small`, `unet_large`,
        `unet_xl`)
    -   Added support for training with prefiltered auxiliary features by
        passing `--aux_results` to `preprocess.py` and `train.py`
    -   Added experimental support for depth (`z`)

### Changes in v2.2.2:

-   Fully fixed GPU memory leak when releasing SYCL, CUDA and HIP device objects
-   Fixed CUDA context error in some cases when using the CUDA driver API
-   Fixed crash on systems with an unsupported AMD Vega integrated GPU and old
    driver

### Changes in v2.2.1:

-   Fixed memory leak when releasing SYCL, CUDA and HIP device objects
-   Fixed memory leak when initializing Metal filters

### Changes in v2.2.0:

-   Improved denoising quality (better fine detail reconstruction)
-   Added Intel Meteor Lake GPU support (in Intel® Core™ Ultra Processors)
-   Added Metal device for Apple silicon GPUs (requires macOS Ventura or newer)
-   Added ARM64 (AArch64) CPU support on Windows and Linux (in addition to macOS)
-   Improved CPU performance
-   Significantly reduced overhead of committing filter changes
-   Switched to the CUDA driver API by default, added the `OIDN_DEVICE_CUDA_API`
    CMake option for manually selecting between the driver and runtime APIs
-   Fixed crash when releasing a buffer after releasing the device

### Changes in v2.1.0:

-   Added support for denoising 1-channel (e.g. alpha) and 2-channel images
-   Added support for arbitrary combinations of input image data types
    (e.g. `OIDN_FORMAT_FLOAT3` for `color` but `OIDN_FORMAT_HALF3` for `albedo`)
-   Improved performance for most dedicated GPU architectures
-   Re-added `OIDN_STATIC_LIB` CMake option which enables building as a static
    (CPU support only) or a hybrid static/shared (GPU support as well) library
-   Added `release()` method to C++ API objects (`DeviceRef`, `BufferRef`,
    `FilterRef`)
-   Fixed possible crash when releasing GPU devices, buffers or filters
-   Fixed possible crash at process exit for some SYCL runtime versions
-   Fixed image quality inconsistency on Intel integrated GPUs, but at the cost
    of some performance loss
-   Fixed future Windows driver compatibility for Intel integrated GPUs
-   Fixed rare output corruption on AMD RDNA2 GPUs
-   Fixed device detection on Windows when the path to the library has non-ANSI
    characters
-   Added support for Intel® oneAPI DPC++/C++ Compiler 2024.0 and compatible
    open source compiler versions
-   Upgraded to oneTBB 2021.10.0 in the official binaries
-   Improved detection of old oneTBB versions

### Changes in v2.0.1:

-   Fixed performance issue for Intel integrated GPUs using recent Linux drivers
-   Fixed crash on systems with both dedicated and integrated AMD GPUs
-   Fixed importing `D3D12_RESOURCE`, `D3D11_RESOURCE`, `D3D11_RESOURCE_KMT`,
    `D3D11_TEXTURE` and `D3D11_TEXTURE_KMT` external memory types on CUDA and
    HIP devices
-   Fixed the macOS deployment target of the official x86 binaries (lowered from
    11.0 to 10.11)
-   Minor improvements to verbose output

### Changes in v2.0.0:

-   Added SYCL device for Intel Xe architecture GPUs (Xe-LP, Xe-HPG and Xe-HPC)
-   Added CUDA device for NVIDIA Volta, Turing, Ampere, Ada Lovelace and Hopper
    architecture GPUs
-   Added HIP device for AMD RDNA2 (Navi 21 only) and RDNA3 (Navi 3x)
    architecture GPUs
-   Added new buffer API functions for specifying the storage type (host, device
    or managed), copying data to/from the host, and importing external buffers from
    graphics APIs (e.g. Vulkan, Direct3D 12)
-   Removed the `oidnMapBuffer` and `oidnUnmapBuffer` functions
-   Added support for asynchronous execution (e.g. `oidnExecuteFilterAsync`,
    `oidnSyncDevice` functions)
-   Added physical device API for querying the supported devices in the system
-   Added functions for creating a device from a physical device ID, UUID, LUID
    or PCI address (e.g. `oidnNewDeviceByID`)
-   Added SYCL, CUDA and HIP interoperability API functions (e.g. `oidnNewSYCLDevice`,
    `oidnExecuteSYCLFilterAsync`)
-   Added `type` device parameter for querying the device type
-   Added `systemMemorySupported` and `managedMemorySupported` device parameters
    for querying memory allocations supported by the device
-   Added `externalMemoryTypes` device parameter for querying the supported
    external memory handle types
-   Added `quality` filter parameter for setting the filtering quality mode (*high*
    or *balanced* quality)
-   Minor API changes with backward compatibility:
    -   Added `oidn(Get|Set)(Device|Filter)(Bool|Int|Float)` functions and
        deprecated `oidn(Get|Set)(Device|Filter)(1b|1i|1f)` functions
    -   Added `oidnUnsetFilter(Image|Data)` functions and deprecated
        `oidnRemoveFilter(Image|Data)` functions
    -   Renamed `alignment` and `overlap` filter parameters to `tileAlignment`
        and `tileOverlap` but the old names remain supported
-   Removed `OIDN_STATIC_LIB` and `OIDN_STATIC_RUNTIME` CMake options due to
    technical limitations
-   Fixed over-conservative buffer bounds checking for images with custom strides
-   Upgraded to oneTBB 2021.9.0 in the official binaries

### Changes in v1.4.3:

-   Fixed hardcoded library paths in installed macOS binaries
-   Disabled VTune profiling support of oneDNN kernels by default, can be
    enabled using CMake options if required (`DNNL_ENABLE_JIT_PROFILING` and
    `DNNL_ENABLE_ITT_TASKS`)
-   Upgraded to oneTBB 2021.5.0 in the official binaries

### Changes in v1.4.2:

-   Added support for 16-bit half-precision floating-point images
-   Added `oidnGetBufferData` and `oidnGetBufferSize` functions
-   Fixed performance issue on x86 hybrid architecture CPUs (e.g. Alder Lake)
-   Fixed build error when using OpenImageIO 2.3 or later
-   Upgraded to oneTBB 2021.4.0 in the official binaries

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
