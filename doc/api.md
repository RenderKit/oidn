Open Image Denoise API
======================

Open Image Denoise provides a C99 API (also compatible with C++) and a C++11
wrapper API as well. For simplicity, this document mostly refers to the C99
version of the API.

The API is designed in an object-oriented manner, e.g. it contains device
objects (`OIDNDevice` type), buffer objects (`OIDNBuffer` type), and filter
objects (`OIDNFilter` type). All objects are reference-counted, and handles
can be released by calling the appropriate release function (e.g.
`oidnReleaseDevice`) or retained by incrementing the reference count (e.g.
`oidnRetainDevice`).

An important aspect of objects is that setting their parameters do not have
an immediate effect (with a few exceptions). Instead, objects with updated
parameters are in an unusable state until the parameters get explicitly
committed to a given object. The commit semantic allows for batching up
multiple small changes, and specifies exactly when changes to objects will
occur.

All API calls are thread-safe, but operations that use the same device will be
serialized, so the amount of API calls from different threads should be minimized.

Examples
--------

To have a quick overview of the C99 and C++11 APIs, see the following
simple example code snippets.

### Basic Denoising (C99 API)

    #include <OpenImageDenoise/oidn.h>
    ...

    // Create an Open Image Denoise device
    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT); // CPU or GPU if available
    // OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
    oidnCommitDevice(device);

    // Create buffers for input/output images accessible by both host (CPU) and device (CPU/GPU)
    OIDNBuffer colorBuf  = oidnNewBuffer(device, width * height * 3 * sizeof(float));
    OIDNBuffer albedoBuf = ...

    // Create a filter for denoising a beauty (color) image using optional auxiliary images too
    // This can be an expensive operation, so try not to create a new filter for every image!
    OIDNFilter filter = oidnNewFilter(device, "RT"); // generic ray tracing filter
    oidnSetFilterImage(filter, "color",  colorBuf,
                       OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // beauty
    oidnSetFilterImage(filter, "albedo", albedoBuf,
                       OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // auxiliary
    oidnSetFilterImage(filter, "normal", normalBuf,
                       OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // auxiliary
    oidnSetFilterImage(filter, "output", colorBuf,
                       OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // denoised beauty
    oidnSetFilterBool(filter, "hdr", true); // beauty image is HDR
    oidnCommitFilter(filter);

    // Fill the input image buffers
    float* colorPtr = (float*)oidnGetBufferData(colorBuf);
    ...

    // Filter the beauty image
    oidnExecuteFilter(filter);

    // Check for errors
    const char* errorMessage;
    if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
      printf("Error: %s\n", errorMessage);

    // Cleanup
    oidnReleaseBuffer(colorBuf);
    ...
    oidnReleaseFilter(filter);
    oidnReleaseDevice(device);

### Basic Denoising (C++11 API)

    #include <OpenImageDenoise/oidn.hpp>
    ...

    // Create an Open Image Denoise device
    oidn::DeviceRef device = oidn::newDevice(); // CPU or GPU if available
    // oidn::DeviceRef device = oidn::newDevice(oidn::DeviceType::CPU);
    device.commit();

    // Create buffers for input/output images accessible by both host (CPU) and device (CPU/GPU)
    oidn::BufferRef colorBuf  = device.newBuffer(width * height * 3 * sizeof(float));
    oidn::BufferRef albedoBuf = ...

    // Create a filter for denoising a beauty (color) image using optional auxiliary images too
    // This can be an expensive operation, so try no to create a new filter for every image!
    oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
    filter.setImage("color",  colorBuf,  oidn::Format::Float3, width, height); // beauty
    filter.setImage("albedo", albedoBuf, oidn::Format::Float3, width, height); // auxiliary
    filter.setImage("normal", normalBuf, oidn::Format::Float3, width, height); // auxiliary
    filter.setImage("output", colorBuf,  oidn::Format::Float3, width, height); // denoised beauty
    filter.set("hdr", true); // beauty image is HDR
    filter.commit();

    // Fill the input image buffers
    float* colorPtr = (float*)colorBuf.getData();
    ...

    // Filter the beauty image
    filter.execute();

    // Check for errors
    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
      std::cout << "Error: " << errorMessage << std::endl;

### Denoising with Prefiltering (C++11 API)

    // Create a filter for denoising a beauty (color) image using prefiltered auxiliary images too
    oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
    filter.setImage("color",  colorBuf,  oidn::Format::Float3, width, height); // beauty
    filter.setImage("albedo", albedoBuf, oidn::Format::Float3, width, height); // auxiliary
    filter.setImage("normal", normalBuf, oidn::Format::Float3, width, height); // auxiliary
    filter.setImage("output", outputBuf, oidn::Format::Float3, width, height); // denoised beauty
    filter.set("hdr", true); // beauty image is HDR
    filter.set("cleanAux", true); // auxiliary images will be prefiltered
    filter.commit();

    // Create a separate filter for denoising an auxiliary albedo image (in-place)
    oidn::FilterRef albedoFilter = device.newFilter("RT"); // same filter type as for beauty
    albedoFilter.setImage("albedo", albedoBuf, oidn::Format::Float3, width, height);
    albedoFilter.setImage("output", albedoBuf, oidn::Format::Float3, width, height);
    albedoFilter.commit();

    // Create a separate filter for denoising an auxiliary normal image (in-place)
    oidn::FilterRef normalFilter = device.newFilter("RT"); // same filter type as for beauty
    normalFilter.setImage("normal", normalBuf, oidn::Format::Float3, width, height);
    normalFilter.setImage("output", normalBuf, oidn::Format::Float3, width, height);
    normalFilter.commit();

    // Prefilter the auxiliary images
    albedoFilter.execute();
    normalFilter.execute();

    // Filter the beauty image
    filter.execute();


Upgrading from Open Image Denoise 1.x
-------------------------------------

Open Image Denoise 2 introduces GPU support, which requires implementing some
minor changes in applications. There are also small API changes, additions and
improvements in this new version. In this section we summarize the necessary
code modifications and also briefly mention the new features that users might
find useful when upgrading to version 2.x. For a full description of the changes
and new functionality, please see the API reference.

### Buffers

The most important required change is related to how data is passed to Open
Image Denoise. If the application is explicitly using only the CPU (by
specifying `OIDN_DEVICE_TYPE_CPU`), no changes should be necessary. But if it
wants to support GPUs as well, passing pointers to memory allocated with the
system allocator (e.g. `malloc`) would raise an error because GPUs cannot access
such memory in almost all cases.

To ensure compatibility with any kind of device, including GPUs, the application
should use `OIDNBuffer` objects to store all image data passed to the library.
Memory allocated using buffers is by default accessible by both the host (CPU)
and the device (CPU or GPU).

Ideally, the application should directly read and write image data to/from
such buffers to avoid redundant and inefficient data copying. If this cannot be
implemented, the application should try to minimize the overhead of copying as
much as possible:

-   Data should be copied to/from buffers only if the data in system memory
    indeed cannot be accessed by the device. This can be determined by simply
    querying the `systemMemorySupported` device parameter. If system memory is
    accessible by the device, no buffers are necessary and filter image
    parameters can be set with `oidnSetSharedFilterImage`.

-   If the image data cannot be accessed by the device, buffers must be created
    and the data must be copied to/from these buffers. These buffers should be
    directly passed to filters as image parameters instead of the original
    pointers using `oidnSetFilterImage`.

-   Data should be copied asynchronously using using the new
    `oidnReadBufferAsync` and `oidnWriteBufferAsync` functions, which may
    achieve higher performance than plain `memcpy`.

-   If image data must be copied, using the default buffer allocation may not be
    the most efficient method. If the device memory is not physically shared
    with the host memory (e.g. for dedicated GPUs), higher performance may be
    achieved by creating the buffers with device storage (`OIDN_STORAGE_DEVICE`)
    using the new `oidnNewBufferWithStorage` function. This way, the buffer data
    cannot be directly accessed by the host anymore but this should not matter
    because the data must be copied from some other memory location anyway.
    However, this ensures that the data is stored only in high-performance
    device memory, and the user has full control over when and how the data is
    transferred between host and device.

The `oidnMapBuffer` and `oidnUnmapBuffer` functions have been removed from the
API due to these not being supported by any of the device backends. Please use
`oidnReadBuffer(Async)` and `oidnWriteBuffer(Async)` instead.

### Interop with Compute (SYCL, CUDA, HIP) and Graphics (DX, Vulkan, Metal) APIs

If the application is explicitly using a particular device type which supports
unified memory allocations, e.g. SYCL or CUDA, it may directly pass pointers
allocated using the native allocator of the respective compute API (e.g.
`sycl::malloc_device`, `cudaMalloc`) instead of using buffers. This way, it is
the responsibility of the user to correctly allocate the memory for the device.

In such cases, it often necessary to have more control over the device creation
as well, to ensure that filtering is running on the intended device and command
queues or streams from the application can be shared to improve performance. If
the application is using the same compute or graphics API as the Open Image
Denoise device, this can be achieved by creating devices with
`oidnNewSYCLDevice`, `oidnNewCUDADevice`, etc. For some APIs there are
additional interoperability functions as well, e.g. `oidnExecuteSYCLFilterAsync`.

If the application is using a graphics API which does not support unified memory
allocations, e.g. DX12 or Vulkan, it may be still possible to share memory
between the application and Open Image Denoise using buffers, avoiding
expensive copying through host memory. External buffers can be imported from
graphics APIs with the new `oidnNewSharedBufferFromFD` and
`oidnNewSharedBufferFromWin32Handle` functions. To use this feature, buffers
must be exported in the graphics API and must be imported in Open Image Denoise
using the same kind of handle. Care must be taken to select an external
memory handle type which is supported by both APIs. The external memory types
supported by an Open Image Denoise device can be queried using the
`externalMemoryTypes` device parameter. Note that some devices do not support
importing external memory at all (e.g. CPUs, and on GPUs it primarily depends
on the installed drivers), so the application should always implement a fallback
too, which copies the data through the host if there is no other supported way.
Metal buffers can be used directly with the `oidnNewSharedBufferFromMetal`
function.

Sharing textures is currently not supported natively but it is still possible
avoid copying texture data by using a linear texture layout (e.g.
`VK_IMAGE_TILING_LINEAR` in Vulkan) and sharing the buffer that backs this
data. In this case, you should ensure that the row stride of the linear texture
data is correctly set.

Importing external synchronization primitives (e.g. semaphores) from graphics
APIs is not yet supported either but it is planned for a future release.
Meanwhile, synchronizing access to shared memory should be done on the host
using `oidnSyncDevice` and the used graphics API.

When importing external memory, the application also needs to make sure that the
Open Image Denoise device is running on the same *physical* device as the
graphics API. This can be easily achieved by using the new physical device
feature, described in the next section.

### Physical Devices

Although it is possible to explicitly create devices of a particular type (with,
e.g., `OIDN_DEVICE_TYPE_SYCL`), this is often insufficient, especially if
the system has multiple devices of the same type, and with GPU support it is
very common that there are multiple different types of supported devices in the
system (e.g. a CPU and one or more GPUs).

Open Image Denoise 2 introduces a simple *physical device* API, which
enables the application to query the list of supported physical devices in the
system, including their name, type, UUID, LUID, PCI address, etc. (see
`oidnGetNumPhysicalDevices`, `oidnGetPhysicalDeviceString`, etc.). New logical
device (i.e. `OIDNDevice`) creation functions for have been also introduced, which
enable creating a logical device on a specific physical device:
`oidnNewDeviceByID`, `oidnNewDeviceByUUID`, etc.

Creating a logical device on a physical device having a particular UUID, LUID
or PCI address is particularly important when importing external memory from
graphics APIs. However, not all device types support all types of IDs, and some
graphics drivers may even report mismatching UUIDs or LUIDs for the same
physical device, so applications should try to implement multiple identification
methods, or at least assume that identification might fail.

### Asynchronous Execution

It is now possible to execute some operations asynchronously, most importantly
filtering (`oidnExecuteFilterAsync`, `oidnExecuteSYCLFilterAsync`) and copying
data (the already mentioned `oidnReadBufferAsync` and `oidnWriteBufferAsync`).

When using any asynchronous function it is the responsibility of the
application to handle correct synchronization using `oidnSyncDevice`.

### Filter Quality

Open Image Denoise still delivers the same high image quality on all device
types as before, including on GPUs. But often filtering performance is more
important than having the highest possible image quality, so it is now possible
to switch between multiple filter quality modes. Filters have a new
parameter called `quality`, which defaults to the existing high image quality
(`OIDN_QUALITY_HIGH`) but a balanced quality mode (`OIDN_QUALITY_BALANCED`)
has been added as well for even higher performance. We recommend using balanced
quality for interactive and real-time use cases.

### Small API Changes

A few existing API functions have been renamed to improve clarity (e.g.
`oidnSetFilter1i` to `oidnSetFilterInt`) but the old function names are still
available as deprecated functions. When compiling legacy code, warnings will
be emitted for these deprecated functions. To upgrade to the new API, please
simply follow the instructions in the warnings.

Some filter parameters have been also renamed (`alignment` to `tileAlignment`,
`overlap` to `tileOverlap`). When using the old names, warnings will be emitted
at runtime.

### Building as a Static Library

The support to build Open Image Denoise as a static library (`OIDN_STATIC_LIB`
CMake option) has been limited to CPU-only builds due to switching to a modular
library design that was necessary for adding multi-vendor GPU support. If the
library is built with GPU support as well, the `OIDN_STATIC_LIB` option is still
available but enabling it results in a hybrid static/shared library.

If the main reason for building as a static library would be is the ability to
use multiple versions of Open Image Denoise in the same process, please use the
existing `OIDN_API_NAMESPACE` CMake option instead. With this feature all
symbols of the library will be put into a custom namespace, which can prevent
symbol clashes.


Physical Devices
----------------

Systems often have multiple different types of devices supported by Open
Image Denoise (CPUs and GPUs). The application can get the list of supported
*physical devices* and select which of these to use for denoising.

The number of supported physical devices can be queried with

    int oidnGetNumPhysicalDevices();

The physical devices can be identified using IDs between 0 and
(`oidnGetNumPhysicalDevices()` $-$ 1), and are ordered *approximately* from
fastest to slowest (e.g., ID of 0 corresponds to the likely fastest physical
device). Note that the reported number and order of physical devices may change
between application runs, so no assumptions should be made about this list.

Parameters of these physical devices can be queried using

    bool         oidnGetPhysicalDeviceBool  (int physicalDeviceID, const char* name);
    int          oidnGetPhysicalDeviceInt   (int physicalDeviceID, const char* name);
    unsigned int oidnGetPhysicalDeviceUInt  (int physicalDeviceID, const char* name);
    const char*  oidnGetPhysicalDeviceString(int physicalDeviceID, const char* name);
    const void*  oidnGetPhysicalDeviceData  (int physicalDeviceID, const char* name,
                                             size_t* byteSize);

where `name` is the name of the parameter, and `byteSize` is the number of returned
bytes for data parameters. The following parameters can be queried:

----------- ---------------------  -----------------------------------------------------------------
Type        Name                   Description
----------- ---------------------  -----------------------------------------------------------------
`Int`       `type`                 device type as an `OIDNDeviceType` value

`String`    `name`                 name string

`Bool`      `uuidSupported`        device supports universally unique identifier (UUID)

`Data`      `uuid`                 opaque UUID (`OIDN_UUID_SIZE` bytes, exists only if
                                   `uuidSupported` is `true`)

`Bool`      `luidSupported`        device supports locally unique identifier (UUID)

`Data`      `luid`                 opaque LUID (`OIDN_LUID_SIZE` bytes, exists only if
                                   `luidSupported` is `true`)

`UInt`      `nodeMask`             bitfield identifying the node within a linked device adapter
                                   corresponding to the device (exists only if `luidSupported` is
                                   `true`)

`Bool`      `pciAddressSupported`  device supports PCI address

`Int`       `pciDomain`            PCI domain (exists only if `pciAddressSupported` is `true`)

`Int`       `pciBus`               PCI bus (exists only if `pciAddressSupported` is `true`)

`Int`       `pciDevice`            PCI device (exists only if `pciAddressSupported` is `true`)

`Int`       `pciFunction`          PCI function (exists only if `pciAddressSupported` is `true`)
----------- ---------------------  -----------------------------------------------------------------
: Constant parameters supported by physical devices.


Devices
-------

Open Image Denoise has a *logical* device concept as well, or simply referred
to as *device*, which allows different components of the application to use the
Open Image Denoise API without interfering with each other. Each physical device
may be associated with one ore more logical devices. A basic way to create a
device is by calling

    OIDNDevice oidnNewDevice(OIDNDeviceType type);

where the `type` enumeration maps to a specific device implementation, which
can be one of the following:

-------------------------- --------------------------------------------------------------------
Name                       Description
-------------------------- --------------------------------------------------------------------
`OIDN_DEVICE_TYPE_DEFAULT` select the likely fastest device (same as physical device with ID 0)

`OIDN_DEVICE_TYPE_CPU`     CPU device

`OIDN_DEVICE_TYPE_SYCL`    SYCL device (requires a supported Intel GPU)

`OIDN_DEVICE_TYPE_CUDA`    CUDA device (requires a supported NVIDIA GPU)

`OIDN_DEVICE_TYPE_HIP`     HIP device (requires a supported AMD GPU)

`OIDN_DEVICE_TYPE_METAL`   Metal device (requires a supported Apple GPU)
-------------------------- --------------------------------------------------------------------
: Supported device types, i.e., valid constants of type `OIDNDeviceType`.

If there are multiple supported devices of the specified type, an implementation-dependent
default will be selected.

A device can be created by specifying a physical device ID as well using

    OIDNDevice oidnNewDeviceByID(int physicalDeviceID);

Applications can manually iterate over the list of physical devices and select
from them based on their properties but there are also some built-in helper
functions as well, which make creating a device by a particular physical device
property easier:

    OIDNDevice oidnNewDeviceByUUID(const void* uuid);
    OIDNDevice oidnNewDeviceByLUID(const void* luid);
    OIDNDevice oidnNewDeviceByPCIAddress(int pciDomain, int pciBus, int pciDevice,
                                         int pciFunction);

These functions are particularly useful when the application needs
interoperability with a graphics API (e.g. DX12, Vulkan). However, not all of
these properties may be supported by the intended physical device (or drivers
might even report inconsistent identifiers), so it is recommended to select by
more than one property, if possible.

If the application requires interoperability with a particular compute or
graphics API (SYCL, CUDA, HIP, Metal), it is recommended to use one of the
following dedicated functions instead:

    OIDNDevice oidnNewSYCLDevice(const sycl::queue* queues, int numQueues);
    OIDNDevice oidnNewCUDADevice(const int* deviceIDs, const cudaStream_t* streams,
                                 int numPairs);
    OIDNDevice oidnNewHIPDevice(const int* deviceIDs, const hipStream_t* streams,
                                int numPairs);
    OIDNDevice oidnNewMetalDevice(const MTLCommandQueue_id* commandQueues,
                                  int numQueues);

For SYCL, it is possible to pass one or more SYCL queues which will be used by
Open Image Denoise for all device operations. This is useful when the
application wants to use the same queues for both denoising and its own
operations (e.g. rendering). Passing multiple queues is not intended to be used
for different physical devices but just for a single SYCL root-device which
consists of multiple sub-devices (e.g. Intel® Data Center GPU Max Series having
multiple Xe-Stacks/tiles). The only supported SYCL backend is oneAPI Level Zero.

For CUDA and HIP, pairs of CUDA/HIP device IDs and corresponding streams can be
specified but the current implementation supports only one pair. A `NULL` stream
corresponds to the default stream on the corresponding device. Open Image Denoise
automatically sets and restores the current CUDA/HIP device/context on the
calling thread when necessary, thus the current device does not have to be
changed manually by the application.

For Metal, a single command queue is supported.

Once a device is created, you can call

    bool oidnGetDeviceBool(OIDNDevice device, const char* name);
    void oidnSetDeviceBool(OIDNDevice device, const char* name, bool value);
    int  oidnGetDeviceInt (OIDNDevice device, const char* name);
    void oidnSetDeviceInt (OIDNDevice device, const char* name, int  value);
    int  oidnGetDeviceUInt(OIDNDevice device, const char* name);
    void oidnSetDeviceUInt(OIDNDevice device, const char* name, unsigned int value);

to set and get parameter values on the device. Note that some parameters are
constants, thus trying to set them is an error. See the tables below for the
parameters supported by devices.

----------- ----------------------- ----------- ----------------------------------------------------
Type        Name                        Default Description
----------- ------------------------ ---------- ----------------------------------------------------
`Int`       `type`                   *constant* device type as an `OIDNDeviceType` value

`Int`       `version`                *constant* combined version number (major.minor.patch)
                                                with two decimal digits per component

`Int`       `versionMajor`           *constant* major version number

`Int`       `versionMinor`           *constant* minor version number

`Int`       `versionPatch`           *constant* patch version number

`Bool`      `systemMemorySupported`  *constant* device can directly access memory allocated with the
                                                system allocator (e.g. `malloc`)

`Bool`      `managedMemorySupported` *constant* device supports buffers created with managed storage
                                                (`OIDN_STORAGE_MANAGED`)

`Int`       `externalMemoryTypes`    *constant* bitfield of `OIDNExternalMemoryTypeFlag` values
                                                representing the external memory types supported by
                                                the device

`Int`       `verbose`                         0 verbosity level of the console output between 0--4;
                                                when set to 0, no output is printed, when set to a
                                                higher level more output is printed
----------- ------------------------ ---------- ----------------------------------------------------
: Parameters supported by all devices.

------ -------------- -------- -------------------------------------------------
Type   Name            Default Description
------ -------------- -------- -------------------------------------------------
`Int`  `numThreads`          0 maximum number of threads which the library
                               should use; 0 will set it automatically to get
                               the best performance

`Bool` `setAffinity`    `true` enables thread affinitization (pinning software
                               threads to hardware threads) if it is necessary
                               for achieving optimal performance
------ -------------- -------- -------------------------------------------------
: Additional parameters supported only by CPU devices.

Note that the CPU device heavily relies on setting the thread affinities to
achieve optimal performance, so it is highly recommended to leave this option
enabled. However, this may interfere with the application if that also sets
the thread affinities, potentially causing performance degradation. In such
cases, the recommended solution is to either disable setting the affinities
in the application or in Open Image Denoise, or to always set/reset
the affinities before/after each parallel region in the application (e.g.,
if using TBB, with `tbb::task_arena` and `tbb::task_scheduler_observer`).

Once parameters are set on the created device, the device must be committed with

    void oidnCommitDevice(OIDNDevice device);

This device can then be used to construct further objects, such as buffers and
filters. Note that a device can be committed only once during its lifetime.

If the goal is not to set up a device object for actual use yet but only to
check whether the device with the current parameters is supported, the following
function could be called instead, which does not require committing the device
first (which could be potentially more expensive):

    bool oidnIsDeviceSupported(OIDNDevice device);

Some functions may execute asynchronously with respect to the host. The names of
these functions are suffixed with `Async`. Asynchronous operations are executed
*in order* on the device but may not block on the host. Eventually, it is
necessary to wait for all asynchronous operations to complete, which can be done
by calling

    void oidnSyncDevice(OIDNDevice device);

Before the application exits, it should release all devices by invoking

    void oidnReleaseDevice(OIDNDevice device);

Note that Open Image Denoise uses reference counting for all object types,
so this function decreases the reference count of the device, and if the count
reaches 0 the device will automatically get deleted. It is also possible to
increase the reference count by calling

    void oidnRetainDevice(OIDNDevice device);

An application should typically create only a single device object per physical
device (one for *all* CPUs or one per GPU) as creation can be very expensive and
additional device objects may incur a significant memory overhead. If required
differently, it should only use a small number of device objects at any given
time.

### Error Handling

Each user thread has its own error code per device. If an error occurs when
calling an API function, this error code is set to the occurred error if it
stores no previous error. The currently stored error can be queried by the
application via

    OIDNError oidnGetDeviceError(OIDNDevice device, const char** outMessage);

where `outMessage` can be a pointer to a C string which will be set to a more
descriptive error message, or it can be `NULL`. This function also clears the
error code, which assures that the returned error code is always the first
error occurred since the last invocation of `oidnGetDeviceError` on the current
thread. Note that the optionally returned error message string is valid only
until the next invocation of the function.

Alternatively, the application can also register a callback function of type

    typedef void (*OIDNErrorFunction)(void* userPtr, OIDNError code, const char* message);

via

    void oidnSetDeviceErrorFunction(OIDNDevice device, OIDNErrorFunction func, void* userPtr);

to get notified when errors occur. Only a single callback function can be
registered per device, and further invocations overwrite the previously set
callback function, which do *not* require also calling the `oidnCommitDevice`
function. Passing `NULL` as function pointer disables the registered callback
function. When the registered callback function is invoked, it gets passed the
user-defined payload (`userPtr` argument as specified at registration time),
the error code (`code` argument) of the occurred error, as well as a string
(`message` argument) that further describes the error. The error code is always
set even if an error callback function is registered. It is recommended to
always set a error callback function, to detect all errors.

When the device construction fails, `oidnNewDevice` returns `NULL` as device.
To detect the error code of a such failed device construction, pass `NULL` as
device to the `oidnGetDeviceError` function. For all other invocations of
`oidnGetDeviceError`, a proper device handle must be specified.

The following errors are currently used by Open Image Denoise:

--------------------------------- ----------------------------------------------
Name                              Description
--------------------------------- ----------------------------------------------
`OIDN_ERROR_NONE`                 no error occurred

`OIDN_ERROR_UNKNOWN`              an unknown error occurred

`OIDN_ERROR_INVALID_ARGUMENT`     an invalid argument was specified

`OIDN_ERROR_INVALID_OPERATION`    the operation is not allowed

`OIDN_ERROR_OUT_OF_MEMORY`        not enough memory to execute the operation

`OIDN_ERROR_UNSUPPORTED_HARDWARE` the hardware (CPU/GPU) is not supported

`OIDN_ERROR_CANCELLED`            the operation was cancelled by the user
--------------------------------- ----------------------------------------------
: Possible error codes, i.e., valid constants of type `OIDNError`.

### Environment Variables

Open Image Denoise supports environment variables for overriding certain
settings at runtime, which can be useful for debugging and development:

Name                     Description
------------------------ ---------------------------------------------------------------------------
`OIDN_DEFAULT_DEVICE`    overrides what physical device to use with `OIDN_DEVICE_TYPE_DEFAULT`; can be `cpu`, `sycl`, `cuda`, `hip`, or a physical device ID
`OIDN_DEVICE_CPU`        value of 0 disables CPU device support
`OIDN_DEVICE_SYCL`       value of 0 disables SYCL device support
`OIDN_DEVICE_CUDA`       value of 0 disables CUDA device support
`OIDN_DEVICE_HIP`        value of 0 disables HIP device support
`OIDN_DEVICE_METAL`      value of 0 disables Metal device support
`OIDN_NUM_THREADS`       overrides `numThreads` device parameter
`OIDN_SET_AFFINITY`      overrides `setAffinity` device parameter
`OIDN_NUM_SUBDEVICES`    overrides number of SYCL sub-devices to use (e.g. for Intel® Data Center GPU Max Series)
`OIDN_VERBOSE`           overrides `verbose` device parameter
------------------------ ---------------------------------------------------------------------------
: Environment variables supported by Open Image Denoise.


Buffers
-------

Image data can be passed to Open Image Denoise either via pointers to
memory allocated and managed by the user or by creating buffer objects.
Regardless of which method is used, the data must be allocated in a way that it
is accessible by the device (either CPU or GPU). Using buffers is typically the
preferred approach because this ensures that the allocation requirements are
fulfilled regardless of device type. To create a new data buffer with memory
allocated and owned by the device, use

    OIDNBuffer oidnNewBuffer(OIDNDevice device, size_t byteSize);

The created buffer is bound to the specified device (`device` argument). The
specified number of bytes (`byteSize`) are allocated at buffer construction time
and deallocated when the buffer is destroyed. The memory is by default allocated
as managed memory automatically migrated between host and device, if supported,
or as pinned host memory otherwise.

If this default buffer allocation is not suitable, a buffer can be created with
a manually specified storage mode as well:

    OIDNBuffer oidnNewBufferWithStorage(OIDNDevice device, size_t byteSize, OIDNStorage storage);

The supported storage modes are the following:

------------------------ ---------------------------------------------------------------------------
Name                     Description
------------------------ ---------------------------------------------------------------------------
`OIDN_STORAGE_UNDEFINED` undefined storage mode

`OIDN_STORAGE_HOST`      pinned host memory, accessible by both host and device

`OIDN_STORAGE_DEVICE`    device memory, *not* accessible by the host

`OIDN_STORAGE_MANAGED`   automatically migrated between host and device, accessible by both
                         (*not* supported by all devices, `managedMemorySupported` device parameter
                          must be checked before use)
------------------------ ---------------------------------------------------------------------------
: Supported storage modes for buffers, i.e., valid constants of type `OIDNStorage`.

Note that the host and device storage modes are supported by all devices but managed storage is
an optional feature. Before using managed storage, the `managedMemorySupported` device parameter
should be queried.

It is also possible to create a "shared" data buffer with memory allocated and
managed by the user with

    OIDNBuffer oidnNewSharedBuffer(OIDNDevice device, void* devPtr, size_t byteSize);

where `devPtr` points to user-managed device-accessible memory and `byteSize` is
its size in bytes. At buffer construction time no buffer data is allocated, but
the buffer data provided by the user is used. The buffer data must remain valid
for as long as the buffer may be used, and the user is responsible to free the
buffer data when no longer required. The user must also ensure that the memory is
accessible by the device by using allocation functions supported by the device
(e.g. `sycl::malloc_device`, `cudaMalloc`, `hipMalloc`).

Buffers can be also imported from graphics APIs as external memory, to avoid
expensive copying of data through host memory. Different types of external
memory can be imported from either POSIX file descriptors or Win32 handles
using

    OIDNBuffer oidnNewSharedBufferFromFD(OIDNDevice device,
                                         OIDNExternalMemoryTypeFlag fdType,
                                         int fd, size_t byteSize);

    OIDNBuffer oidnNewSharedBufferFromWin32Handle(OIDNDevice device,
                                                  OIDNExternalMemoryTypeFlag handleType,
                                                  void* handle, const void* name, size_t byteSize);

Before exporting memory from the graphics API, the application should find a
handle type which is supported by both the Open Image Denoise device
(see `externalMemoryTypes` device parameter) and the graphics API. Note that
different GPU vendors may support different handle types. To ensure compatibility
with all device types, applications should support at least
`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD` on Windows and both
`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD` and
`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF` on Linux. All possible external memory
types are listed in the table below.

--------------------------------------------------- ----------------------------------------------------------
Name                                                Description
--------------------------------------------------- ----------------------------------------------------------
`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_NONE`

`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD`          opaque POSIX file descriptor handle (recommended on Linux)

`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF`            file descriptor handle for a Linux dma_buf (recommended on Linux)

`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32`       NT handle (recommended on Windows)

`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT`   global share (KMT) handle

`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE`      NT handle returned by `IDXGIResource1::CreateSharedHandle`
                                                    referring to a Direct3D 11 texture resource

`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE_KMT`  global share (KMT) handle returned by
                                                    `IDXGIResource::GetSharedHandle` referring to a Direct3D 11
                                                    texture resource

`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_RESOURCE`     NT handle returned by `IDXGIResource1::CreateSharedHandle`
                                                    referring to a Direct3D 11 resource

`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_RESOURCE_KMT` global share (KMT) handle returned by
                                                    `IDXGIResource::GetSharedHandle` referring to a Direct3D 11
                                                    resource

`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_HEAP`         NT handle returned by `ID3D12Device::CreateSharedHandle`
                                                    referring to a Direct3D 12 heap resource

`OIDN_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_RESOURCE`     NT handle returned by `ID3D12Device::CreateSharedHandle`
                                                    referring to a Direct3D 12 committed resource
--------------------------------------------------- ----------------------------------------------------------
: Supported external memory type flags, i.e., valid constants of type `OIDNExternalMemoryTypeFlag`.

Metal buffers can be imported directly with

    OIDNBuffer oidnNewSharedBufferFromMetal(OIDNDevice device, MTLBuffer_id buffer);

Note that if a buffer with an `MTLStorageModeManaged` storage mode is imported, it is the
responsibility of the user to synchronize the contents of the buffer between the
host and the device.

Similar to device objects, buffer objects are also reference-counted and can be
retained and released by calling the following functions:

    void oidnRetainBuffer (OIDNBuffer buffer);
    void oidnReleaseBuffer(OIDNBuffer buffer);

The size of in bytes and storage mode of the buffer can be queried using

    size_t      oidnGetBufferSize   (OIDNBuffer buffer);
    OIDNStorage oidnGetBufferStorage(OIDNBuffer buffer);

It is possible to get a pointer directly to the buffer data, which is usually
the preferred way to access the data stored in the buffer:

    void* oidnGetBufferData(OIDNBuffer buffer);

Accessing the data on the host through this pointer is possible *only* if the
buffer was created with `OIDN_STORAGE_HOST` or `OIDN_STORAGE_MANAGED`. Note
that a `NULL` pointer may be returned if the buffer is empty.

In some cases better performance can be achieved by using device storage for
buffers. Such data can be accessed on the host by copying to/from host memory
(including pageable system memory) using the following functions:

    void oidnReadBuffer(OIDNBuffer buffer,
                        size_t byteOffset, size_t byteSize, void* dstHostPtr);

    void oidnWriteBuffer(OIDNBuffer buffer,
                         size_t byteOffset, size_t byteSize, const void* srcHostPtr);

These functions will always block until the read/write operation has been
completed, which is often suboptimal. The following functions execute these
operations asynchronously:

    void oidnReadBufferAsync(OIDNBuffer buffer,
                             size_t byteOffset, size_t byteSize, void* dstHostPtr);

    void oidnWriteBufferAsync(OIDNBuffer buffer,
                              size_t byteOffset, size_t byteSize, const void* srcHostPtr);

When copying asynchronously, the user must ensure correct synchronization with
the device by calling `oidnSyncDevice` before accessing the copied data or
releasing the buffer. Failure to do so will result in undefined behavior.

### Data Format

Buffers store opaque data and thus have no information about the type and
format of the data. Other objects, e.g. filters, typically require specifying
the format of the data stored in buffers or shared via pointers. This can be
done using the `OIDNFormat` enumeration type:

Name                     Description
------------------------ -------------------------------------------------------
`OIDN_FORMAT_UNDEFINED`  undefined format
`OIDN_FORMAT_FLOAT`      32-bit floating-point scalar
`OIDN_FORMAT_FLOAT[234]` 32-bit floating-point [234]-element vector
`OIDN_FORMAT_HALF`       16-bit floating-point scalar
`OIDN_FORMAT_HALF[234]`  16-bit floating-point [234]-element vector
------------------------ -------------------------------------------------------
: Supported data formats, i.e., valid constants of type `OIDNFormat`.


Filters
-------

Filters are the main objects in Open Image Denoise that are responsible
for the actual denoising. The library ships with a collection of filters which
are optimized for different types of images and use cases. To create a filter
object, call

    OIDNFilter oidnNewFilter(OIDNDevice device, const char* type);

where `type` is the name of the filter type to create. The supported filter
types are documented later in this section.

Creating filter objects can be very expensive, therefore it is *strongly*
recommended to reuse the same filter for denoising as many images as possible,
as long as the these images have the same same size, format, and features (i.e.,
only the memory locations and pixel values may be different). Otherwise (e.g.
for images with different resolutions), reusing the same filter would not have
any benefits.

Once created, filter objects can be retained and released with

    void oidnRetainFilter (OIDNFilter filter);
    void oidnReleaseFilter(OIDNFilter filter);

After creating a filter, it needs to be set up by specifying the input and
output images, and potentially setting other parameter values as well.

To set image parameters of a filter, you can use one of the following functions:

    void oidnSetFilterImage(OIDNFilter filter, const char* name,
                            OIDNBuffer buffer, OIDNFormat format,
                            size_t width, size_t height,
                            size_t byteOffset,
                            size_t pixelByteStride, size_t rowByteStride);

    void oidnSetSharedFilterImage(OIDNFilter filter, const char* name,
                                  void* devPtr, OIDNFormat format,
                                  size_t width, size_t height,
                                  size_t byteOffset,
                                  size_t pixelByteStride, size_t rowByteStride);

It is possible to specify either a data buffer object (`buffer` argument) with
the `oidnSetFilterImage` function, or directly a pointer to user-managed
device-accessible data (`devPtr` argument) with the `oidnSetSharedFilterImage`
function. Regardless of whether a buffer or a pointer is specified, the data
*must* be accessible to the device. The easiest way to guarantee this regardless
of the device type (CPU or GPU) is using buffer objects.

In both cases, you must also specify the name of the image parameter to set
(`name` argument, e.g. `"color"`, `"output"`), the pixel format (`format`
argument), the width and height of the image in number of pixels (`width` and
`height` arguments), the starting offset of the image data (`byteOffset`
argument), the pixel stride (`pixelByteStride` argument) and the row stride
(`rowByteStride` argument), in number of bytes.

If the pixels and/or rows are stored contiguously (tightly packed without any
gaps), you can set `pixelByteStride` and/or `rowByteStride` to 0 to let the
library compute the actual strides automatically, as a convenience.

Images support only `FLOAT` and `HALF` pixel formats with up to 3 channels.
Custom image layouts with extra channels (e.g. alpha channel) or other data are
supported as well by specifying a non-zero pixel stride. This way, expensive
image layout conversion and copying can be avoided but the extra channels will
be ignored by the filter. If these channels also need to be denoised, separate
filters can be used.

To unset a previously set image parameter, returning it to a state as if it had
not been set, call

    void oidnRemoveFilterImage(OIDNFilter filter, const char* name);

Some special data used by filters are opaque/untyped (e.g. trained model weights
blobs), which can be specified with the `oidnSetSharedFilterData` function:

    void oidnSetSharedFilterData(OIDNFilter filter, const char* name,
                                 void* hostPtr, size_t byteSize);

This data (`hostPtr`) must be accessible to the *host*, therefore system memory
allocation is suitable (i.e., there is no reason to use buffer objects for
allocation).

Modifying the contents of an opaque data parameter after setting it as a filter
parameter is allowed but the filter needs to be notified that the data has been
updated by calling

    void oidnUpdateFilterData(OIDNFilter filter, const char* name);

Unsetting an opaque data parameter can be performed with

    void oidnRemoveFilterData(OIDNFilter filter, const char* name);

Filters may have parameters other than buffers as well, which you can set and
get using the following functions:

    bool  oidnGetFilterBool (OIDNFilter filter, const char* name);
    void  oidnSetFilterBool (OIDNFilter filter, const char* name, bool  value);
    int   oidnGetFilterInt  (OIDNFilter filter, const char* name);
    void  oidnSetFilterInt  (OIDNFilter filter, const char* name, int   value);
    float oidnGetFilterFloat(OIDNFilter filter, const char* name);
    void  oidnSetFilterFloat(OIDNFilter filter, const char* name, float value);

Filters support a progress monitor callback mechanism that can be used to report
progress of filter operations and to cancel them as well. Calling
`oidnSetFilterProgressMonitorFunction` registers a progress monitor callback
function (`func` argument) with payload (`userPtr` argument) for the specified
filter (`filter` argument):

    typedef bool (*OIDNProgressMonitorFunction)(void* userPtr, double n);

    void oidnSetFilterProgressMonitorFunction(OIDNFilter filter,
                                              OIDNProgressMonitorFunction func,
                                              void* userPtr);

Only a single callback function can be registered per filter, and further
invocations overwrite the previously set callback function. Passing `NULL` as
function pointer disables the registered callback function. Once registered,
Open Image Denoise will invoke the callback function multiple times during
filter operations, by passing the payload as set at registration time
(`userPtr` argument), and a `double` in the range [0, 1] which estimates the
progress of the operation (`n` argument). When returning `true` from the
callback function, Open Image Denoise will continue the filter operation
normally. When returning `false`, the library will attempt to cancel the filter
operation as soon as possible, and if that is fulfilled, it will raise an
`OIDN_ERROR_CANCELLED` error.

Please note that using a progress monitor callback function introduces some
overhead, which may be significant on GPU devices, hurting performance.
Therefore we recommend progress monitoring only for offline denoising, when
denoising an image is expected to take several seconds.

After setting all necessary parameters for the filter, the changes must be
committed by calling

    void oidnCommitFilter(OIDNFilter filter);

The parameters can be updated after committing the filter, but it must be
re-committed for any new changes to take effect. Committing major changes to the
filter (e.g. setting new image parameters, changing the image resolution) can
be expensive, and thus should not be done frequently (e.g. per frame).

Finally, an image can be filtered by executing the filter with

    void oidnExecuteFilter(OIDNFilter filter);

which will read the input image data from the specified buffers and produce the
denoised output image.

This function will always block until the filtering operation has been completed.
The following function executes the operation asynchronously:

    void oidnExecuteFilterAsync(OIDNFilter filter);

For filters created on a SYCL device it is also possible to specify dependent
SYCL events (`depEvents` and `numDepEvents` arguments, may be `NULL`/0) and get
a completion event as well (`doneEvent` argument, may be `NULL`):

    void oidnExecuteSYCLFilterAsync(OIDNFilter filter,
                                    const sycl::event* depEvents, int numDepEvents,
                                    sycl::event* doneEvent);

When filtering asynchronously, the user must ensure correct synchronization with
the device by calling `oidnSyncDevice` before accessing the output image data or
releasing the filter. Failure to do so will result in undefined behavior.

In the following we describe the different filters that are currently
implemented in Open Image Denoise.

### RT

The `RT` (**r**ay **t**racing) filter is a generic ray tracing denoising filter
which is suitable for denoising images rendered with Monte Carlo ray tracing
methods like unidirectional and bidirectional path tracing. It supports depth
of field and motion blur as well, but it is *not* temporally stable. The filter
is based on a convolutional neural network (CNN) and comes with a set of
pre-trained  models that work well with a wide range of ray tracing based
renderers and noise levels.

![Example noisy beauty image rendered using unidirectional path tracing (4
samples per pixel). *Scene by Evermotion.*][imgMazdaColor]

![Example output beauty image denoised using prefiltered auxiliary feature
images (albedo and normal) too.][imgMazdaDenoised]

For denoising *beauty* images, it accepts either a low dynamic range (LDR) or
high dynamic range (HDR) image (`color`) as the main input image. In addition
to this, it also accepts *auxiliary feature* images, `albedo` and `normal`,
which are optional inputs that usually improve the denoising quality
significantly, preserving more details.

It is possible to denoise auxiliary images as well, in which case only the
respective auxiliary image has to be specified as input, instead of the beauty
image. This can be done as a *prefiltering* step to further improve the quality
of the denoised beauty image.

The `RT` filter has certain limitations regarding the supported input images.
Most notably, it cannot denoise images that were not rendered with ray tracing.
Another important limitation is related to anti-aliasing filters. Most
renderers use a high-quality pixel reconstruction filter instead of a trivial
box filter to minimize aliasing artifacts (e.g. Gaussian, Blackman-Harris). The
`RT` filter does support such pixel filters but only if implemented with
importance sampling. Weighted pixel sampling (sometimes called *splatting*)
introduces correlation between neighboring pixels, which causes the denoising
to fail (the noise will not be filtered), thus it is not supported.

The filter can be created by passing `"RT"` to the `oidnNewFilter` function
as the filter type. The filter supports the parameters listed in the table
below. All specified images must have the same dimensions. The output image
can be one of the input images (i.e. in-place denoising is supported). See
section [Examples] for simple code snippets that demonstrate the usage of
the filter.

----------- --------------- ---------- ---------------------------------------------------------------
Type        Name               Default Description
----------- --------------- ---------- ---------------------------------------------------------------
`Image`     `color`         *optional* input beauty image (1--3 channels, LDR values in [0, 1] or HDR
                                       values in [0, +∞), values being interpreted such that, after
                                       scaling with the `inputScale` parameter, a value of 1
                                       corresponds to a luminance level of 100 cd/m²)

`Image`     `albedo`        *optional* input auxiliary image containing the albedo per pixel (1--3
                                       channels, values in [0, 1])

`Image`     `normal`        *optional* input auxiliary image containing the shading normal per pixel
                                       (1--3 channels, world-space or view-space vectors with arbitrary
                                       length, values in [-1, 1])

`Image`     `output`        *required* output image (1--3 channels); can be one of the input images

`Bool`      `hdr`              `false` the main input image is HDR

`Bool`      `srgb`             `false` the main input image is encoded with the sRGB (or 2.2 gamma)
                                       curve (LDR only) or is linear; the output will be encoded
                                       with the same curve

`Float`     `inputScale`           NaN scales values in the main input image before filtering, without
                                       scaling the output too, which can be used to map color or
                                       auxiliary feature values to the expected range, e.g. for
                                       mapping HDR values to physical units (which affects the quality
                                       of the output but *not* the range of the output values); if set
                                       to NaN, the scale is computed implicitly for HDR images or set
                                       to 1 otherwise

`Bool`      `cleanAux`         `false` the auxiliary feature (albedo, normal) images are noise-free;
                                       recommended for highest quality but should *not* be enabled for
                                       noisy auxiliary images to avoid residual noise

`Int`       `quality`             high image quality mode as an `OIDNQuality` value

`Data`      `weights`       *optional* trained model weights blob

`Int`       `maxMemoryMB`           -1 if set to >= 0, a request is made to limit the memory usage
                                       below the specified amount in megabytes at the potential cost
                                       of slower performance, but actual memory usage may be higher
                                       (the target may not be achievable or there may be additional
                                       allocations beyond the control of the library); otherwise,
                                       memory usage will be limited to an unspecified device-dependent
                                       amount; in both cases, filters on the same device share almost
                                       all of their allocated memory to minimize total memory usage

`Int`       `tileAlignment` *constant* when manually denoising in tiles, the tile size and offsets
                                       should be multiples of this amount of pixels to avoid
                                       artifacts; when denoising HDR images `inputScale` *must* be set
                                       by the user to avoid seam artifacts

`Int`       `tileOverlap`   *constant* when manually denoising in tiles, the tiles should overlap by
                                       this amount of pixels

----------- --------------- ---------- ---------------------------------------------------------------
: Parameters supported by the `RT` filter.

Using auxiliary feature images like albedo and normal helps preserving fine
details and textures in the image thus can significantly improve denoising
quality. These images should typically contain feature values for the first
hit (i.e. the surface which is directly visible) per pixel. This works well for
most surfaces but does not provide any benefits for reflections and objects
visible through transparent surfaces (compared to just using the color as
input). However, this issue can be usually fixed by storing feature values for
a subsequent hit (i.e. the reflection and/or refraction) instead of the first
hit. For example, it usually works well to follow perfect specular (*delta*)
paths and store features for the first diffuse or glossy surface hit instead
(e.g. for perfect specular dielectrics and mirrors). This can greatly improve
the quality of reflections and transmission. We will describe this approach in
more detail in the following subsections.

The auxiliary feature images should be as noise-free as possible. It is not a
strict requirement but too much noise in the feature images may cause residual
noise in the output. Ideally, these should be completely noise-free. If this
is the case, this should be hinted to the filter using the `cleanAux` parameter
to ensure the highest possible image quality. But this parameter should be used
with care: if enabled, any noise present in the auxiliary images will end up in
the denoised image as well, as residual noise. Thus, `cleanAux` should be
enabled only if the auxiliary images are guaranteed to be noise-free.

Usually it is difficult to provide clean feature images, and some residual
noise might be present in the output even with `cleanAux` being disabled. To
eliminate this noise and to even improve the sharpness of texture details, the
auxiliary images should be first denoised in a prefiltering step, as mentioned
earlier. Then, these denoised auxiliary images could be used for denoising the
beauty image. Since these are now noise-free, the `cleanAux` parameter should be
enabled. See section [Denoising with prefiltering (C++11 API)] for a simple
code example. Prefiltering makes denoising much more expensive but if there are
multiple color AOVs to denoise, the prefiltered auxiliary images can be reused
for denoising multiple AOVs, amortizing the cost of the prefiltering step.

Thus, for final-frame denoising, where the best possible image quality is
required, it is recommended to prefilter the auxiliary features if they are
noisy and enable the `cleanAux` parameter. Denoising with noisy auxiliary
features should be reserved for previews and interactive rendering.

All auxiliary images should use the same pixel reconstruction filter as the
beauty image. Using a properly anti-aliased beauty image but aliased albedo or
normal images will likely introduce artifacts around edges.

#### Albedos

The albedo image is the feature image that usually provides the biggest quality
improvement. It should contain the approximate color of the surfaces
independent of illumination and viewing angle.

![Example albedo image obtained using the first hit. Note that the albedos of
all transparent surfaces are 1.][imgMazdaAlbedoFirstHit]

![Example albedo image obtained using the first diffuse or glossy (non-delta)
hit. Note that the albedos of perfect specular (delta) transparent surfaces
are computed as the Fresnel blend of the reflected and transmitted
albedos.][imgMazdaAlbedoNonDeltaHit]

For simple matte surfaces this means using the diffuse color/texture as the
albedo. For other, more complex surfaces it is not always obvious what is
the best way to compute the albedo, but the denoising filter is flexible to
a certain extent and works well with differently computed albedos. Thus it is
not necessary to compute the strict, exact albedo values but must be always
between 0 and 1.

For metallic surfaces the albedo should be either the reflectivity at normal
incidence (e.g. from the artist friendly metallic Fresnel model) or the
average reflectivity; or if these are constant (not textured) or unknown, the
albedo can be simply 1 as well.

The albedo for dielectric surfaces (e.g. glass) should be either 1 or, if the
surface is perfect specular (i.e. has a delta BSDF), the Fresnel blend of the
reflected and transmitted albedos. The latter usually works better but only
if it does not introduce too much noise or the albedo is prefiltered. If noise
is an issue, we recommend to split the path into a reflected and a transmitted
path at the first hit, and perhaps fall back to an albedo of 1 for subsequent
dielectric hits. The reflected albedo in itself can be used for mirror-like
surfaces as well.

The albedo for layered surfaces can be computed as the weighted sum of the
albedos of the individual layers. Non-absorbing clear coat layers can be simply
ignored (or the albedo of the perfect specular reflection can be used as well)
but absorption should be taken into account.

#### Normals

The normal image should contain the shading normals of the surfaces either in
world-space or view-space. It is recommended to include normal maps to
preserve as much detail as possible.

![Example normal image obtained using the first hit (the values are actually
in [−1, 1] but were mapped to [0, 1] for illustration
purposes).][imgMazdaNormalFirstHit]

![Example normal image obtained using the first diffuse or glossy (non-delta)
hit. Note that the normals of perfect specular (delta) transparent surfaces
are computed as the Fresnel blend of the reflected and transmitted
normals.][imgMazdaNormalNonDeltaHit]

Just like any other input image, the normal image should be anti-aliased (i.e.
by accumulating the normalized normals per pixel). The final accumulated
normals do not have to be normalized but must be in the [-1, 1] range (i.e.
normals mapped to [0, 1] are *not* acceptable and must be remapped to [−1, 1]).

Similar to the albedo, the normal can be stored for either the first
or a subsequent hit (if the first hit has a perfect specular/delta BSDF).

#### Quality

The filter supports setting an image quality mode, which determines whether to
favor quality, performance, or have a balanced solution between the two. The
supported quality modes are listed in the following table.

Name                     Description
------------------------ ---------------------------------------------------------------------------
`OIDN_QUALITY_DEFAULT`   default quality
`OIDN_QUALITY_BALANCED`  balanced quality/performance (for interactive/real-time rendering)
`OIDN_QUALITY_HIGH`      high quality (for final-frame rendering); *default*
------------------------ ---------------------------------------------------------------------------
: Supported image quality modes, i.e., valid constants of type `OIDNQuality`.

By default, filtering is performed in high quality mode, which is recommended for
final-frame rendering. Using this setting the results have the same high quality
regardless of what kind of device (CPU or GPU) is used. However, due to
significant hardware architecture differences between devices, there might be
small numerical differences between the produced outputs.

The balanced quality mode may provide somewhat lower image quality but higher
performance, and is thus recommended for interactive and real-time rendering.
Note that larger numerical differences should be expected across devices
compared to the high quality mode.

The difference in quality and performance between quality modes depends on the
combination of input features, parameters (e.g. `cleanAux`), and the device
architecture. In some cases the difference may be small or even none.

#### Weights

Instead of using the built-in trained models for filtering, it is also possible
to specify user-trained models at runtime. This can be achieved by passing the
model *weights* blob corresponding to the specified set of features and other
filter parameters, produced by the included training tool. See Section
[Training] for details.

### RTLightmap

The `RTLightmap` filter is a variant of the `RT` filter optimized for denoising
HDR and normalized directional (e.g. spherical harmonics) lightmaps. It does not
support LDR images.

The filter can be created by passing `"RTLightmap"` to the `oidnNewFilter`
function as the filter type. The filter supports the following parameters:

----------- --------------- ---------- ---------------------------------------------------------------
Type        Name               Default Description
----------- --------------- ---------- ---------------------------------------------------------------
`Image`     `color`         *required* input beauty image (1--3 channels, HDR values in [0, +∞),
                                       interpreted such that, after scaling with the `inputScale`
                                       parameter, a value of 1 corresponds to a luminance level of 100
                                       cd/m²; directional values in [-1, 1])

`Image`     `output`        *required* output image (1--3 channels); can be one of the input images

`Bool`      `directional`      `false` whether the input contains normalized coefficients (in [-1, 1])
                                       of a directional lightmap (e.g. normalized L1 or higher
                                       spherical harmonics band with the L0 band divided out); if the
                                       range of the coefficients is different from [-1, 1], the
                                       `inputScale` parameter can be used to adjust the range without
                                       changing the stored values

`Float`     `inputScale`           NaN scales input color values before filtering, without scaling the
                                       output too, which can be used to map color values to the
                                       expected range, e.g. for mapping HDR values to physical units
                                       (which affects the quality of the output but *not* the range of
                                       the output values); if set to NaN, the scale is computed
                                       implicitly for HDR images or set to 1 otherwise

`Int`       `quality`             high image quality mode as an `OIDNQuality` value

`Data`      `weights`       *optional* trained model weights blob

`Int`       `maxMemoryMB`           -1 if set to >= 0, a request is made to limit the memory usage
                                       below the specified amount in megabytes at the potential cost
                                       of slower performance, but actual memory usage may be higher
                                       (the target may not be achievable or there may be additional
                                       allocations beyond the control of the library); otherwise,
                                       memory usage will be limited to an unspecified device-dependent
                                       amount; in both cases, filters on the same device share almost
                                       all of their allocated memory to minimize total memory usage

`Int`       `tileAlignment` *constant* when manually denoising in tiles, the tile size and offsets
                                       should be multiples of this amount of pixels to avoid
                                       artifacts; when denoising HDR images `inputScale` *must* be set
                                       by the user to avoid seam artifacts

`Int`       `tileOverlap`   *constant* when manually denoising in tiles, the tiles should overlap by
                                       this amount of pixels

----------- --------------- ---------- ---------------------------------------------------------------
: Parameters supported by the `RTLightmap` filter.
