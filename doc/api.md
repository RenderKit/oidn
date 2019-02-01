Open Image Denoise API
======================

Open Image Denoise provides a C99 API (also compatible with C++) and a
C++11 wrapper API as well. For simplicity, this document mostly refers to the
C99 version of the API.

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

To have a quick overview of the C99 and C++11 APIs, see the following
simple example code snippets.

### C99 API Example

    #include <OpenImageDenoise/oidn.h>
    ...
    // Create an Open Image Denoise device
    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
    oidnCommitDevice(device);

    // Create a denoising filter
    OIDNFilter filter = oidnNewFilter(device, "RT"); // generic ray tracing filter
    oidnSetSharedFilterImage(filter, "color",  colorPtr, 
                             OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "albedo", albedoPtr,
                             OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // optional
    oidnSetSharedFilterImage(filter, "normal", normalPtr,
                             OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // optional
    oidnSetSharedFilterImage(filter, "output", outputPtr,
                             OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetFilter1b(filter, "hdr", true); // image is HDR
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

### C++11 API Example

    #include <OpenImageDenoise/oidn.hpp>
    ...
    // Create an Open Image Denoise device
    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    // Create a denoising filter
    oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
    filter.setImage("color",  colorPtr,  oidn::Format::Float3, width, height);
    filter.setImage("albedo", albedoPtr, oidn::Format::Float3, width, height); // optional
    filter.setImage("normal", normalPtr, oidn::Format::Float3, width, height); // optional
    filter.setImage("output", outputPtr, oidn::Format::Float3, width, height);
    filter.set("hdr", true); // image is HDR
    filter.commit();

    // Filter the image
    filter.execute();

    // Check for errors
    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
      std::cout << "Error: " << errorMessage << std::endl;


Device
------

Open Image Denoise supports a device concept, which allows different components
of the application to use the Open Image Denoise API without interfering with
each other. An application first needs to create a device with

    OIDNDevice oidnNewDevice(OIDNDeviceType type);

where the `type` enumeration maps to a specific device implementation, which
can be one of the following:

Name                     Description
------------------------ ------------------------------------------------------
OIDN_DEVICE_TYPE_DEFAULT select the approximately fastest device
OIDN_DEVICE_TYPE_CPU     CPU device (requires SSE4.1 support)
------------------------ ------------------------------------------------------
: Supported device types, i.e., valid constants of type `OIDNDeviceType`.

Once a device is created, you can call

    void oidnSetDevice1b(OIDNDevice device, const char* name, bool value);
    void oidnSetDevice1i(OIDNDevice device, const char* name, int  value);
    bool oidnGetDevice1b(OIDNDevice device, const char* name);
    int  oidnGetDevice1i(OIDNDevice device, const char* name);

to set and get parameter values on the device. Note that some parameters are
constants, thus trying to set them is an error. See the tables below for the
parameters supported by devices.

Type      Name         Description
--------- ------------ --------------------------------------------------------
const int version      combined version number (major.minor.patch) with two decimal digits per component
const int versionMajor major version number
const int versionMinor minor version number
const int versionPatch patch version number
--------- ------------ --------------------------------------------------------
: Parameters supported by all devices.

Type   Name          Default Description
------ ------------ -------- --------------------------------------------------
int    numThreads          0 maximum number of threads which Open Image Denoise should use; 0 will set it automatically to get the best performance
bool   setAffinity      true bind software threads to hardware threads if set to true (improves performance); false disables binding
------ ------------ -----------------------------------------------------------
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
Before the application exits, it should release all devices by invoking

    void oidnReleaseDevice(OIDNDevice device);

Note that Open Image Denoise uses reference counting for all object types, so
this function decreases the reference count of the device, and if the count
reaches 0 the device will automatically get deleted. It is also possible to
increase the reference count by calling

    void oidnRetainDevice(OIDNDevice device);

An application typically creates only a single device. If required differently,
it should only use a small number of devices at any given time.

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

------------------------------- -----------------------------------------------
Name                            Description
------------------------------- -----------------------------------------------
OIDN_ERROR_NONE                 no error occurred

OIDN_ERROR_UNKNOWN              an unknown error occurred

OIDN_ERROR_INVALID_ARGUMENT     an invalid argument was specified

OIDN_ERROR_INVALID_OPERATION    the operation is not allowed

OIDN_ERROR_OUT_OF_MEMORY        not enough memory to execute the operation

OIDN_ERROR_UNSUPPORTED_HARDWARE the hardware (e.g., CPU) is not supported
------------------------------- ------------------------------------------------
: Possible error codes, i.e., valid constants of type `OIDNError`.


Buffer
------

Large data like images can be passed to Open Image Denoise either via pointers
to memory allocated and managed by the user (this is the recommended, often
easier and more efficient approach, if supported by the device) or by creating
buffer objects (supported by all devices). To create a new data buffer with
memory allocated and owned by the device, holding `byteSize` number of bytes,
use

    OIDNBuffer oidnNewBuffer(OIDNDevice device, size_t byteSize);

The created buffer is bound to the specified device (`device` argument). The
specified number of bytes are allocated at buffer construction time and
deallocated when the buffer is destroyed.

It is also possible to create a "shared" data buffer with memory allocated and
managed by the user with

    OIDNBuffer oidnNewSharedBuffer(OIDNDevice device, void* ptr, size_t byteSize);

where `ptr` points to the user-managed memory and `byteSize` is its size in
bytes. At buffer construction time no buffer data is allocated, but the buffer
data provided by the user is used. The buffer data must remain valid for as
long as the buffer may be used, and the user is responsible to free the buffer
data when no longer required.

Similar to device objects, buffer objects are also reference-counted and can be
retained and released by calling the following functions:

    void oidnRetainBuffer(OIDNBuffer buffer);
    void oidnReleaseBuffer(OIDNBuffer buffer);

Accessing the data stored in a buffer object is possible by mapping it into the
address space of the application using

    void* oidnMapBuffer(OIDNBuffer buffer, OIDNAccess access, size_t byteOffset, size_t byteSize)

where `access` is the desired access mode of the mapped memory, `byteOffset` is
the offset to the beginning of the mapped memory region in bytes, and
`byteSize` is the number of bytes to map. The function returns a pointer to
the mapped buffer data. If the specified `byteSize` is `0`, the maximum
available amount of memory will be mapped. The `access` argument must be one of
the access modes in the following table:

Name                      Description
------------------------- -----------------------------------------------------
OIDN_ACCESS_READ          read-only access
OIDN_ACCESS_WRITE         write-only access
OIDN_ACCESS_READ_WRITE    read and write access
OIDN_ACCESS_WRITE_DISCARD write-only access but the previous contents will be discarded
------------------------- -----------------------------------------------------
: Access modes for memory regions mapped with `oidnMapBuffer`, i.e., valid
  constants of type `OIDNAccess`.

After accessing the mapped data in the buffer, the memory region must be
unmapped with

    void oidnUnmapBuffer(OIDNBuffer buffer, void* mappedPtr);

where `mappedPtr` must be a pointer returned by a call to `oidnMapBuffer` for
the specified buffer. Any change to the mapped data is guaranteed to take
effect only after unmapping the memory region.

### Data Format

Buffers store opaque data and thus have no information about the type and
format of the data. Other objects, e.g. filters, typically require specifying
the format of the data stored in buffers or shared via pointers. This can be
done using the `OIDNFormat` enumeration type:

Name                   Description
---------------------- --------------------------------------------------------
OIDN_FORMAT_UNDEFINED  undefined format
OIDN_FORMAT_FLOAT      32-bit single-precision floating point scalar
OIDN_FORMAT_FLOAT[234] ... and [234]-element vector
---------------------- --------------------------------------------------------
: Supported data formats, i.e., valid constants of type `OIDNFormat`.


Filter
------

Filters are the main objects in Open Image Denoise that are responsible for the
actual denoising. The library ships with a collection of filters which are
optimized for different types of images and use cases. To create a filter
object, call

    OIDNFilter oidnNewFilter(OIDNDevice device, const char* type);

where `type` is the name of the filter type to create. The supported filter
types are documented later in this section. Once created, filter objects can be
retained and released with

    void oidnRetainFilter(OIDNFilter filter);
    void oidnReleaseFilter(OIDNFilter filter);

After creating a filter, it needs to be set up by specifying the input and
output image buffers, and potentially setting other parameter values as well.

To bind image buffers to the filter, you can use one of the following functions:

    void oidnSetFilterImage(OIDNFilter filter, const char* name,
                            OIDNBuffer buffer, OIDNFormat format,
                            size_t width, size_t height,
                            size_t byteOffset,
                            size_t bytePixelStride, size_t byteRowStride);

    void oidnSetSharedFilterImage(OIDNFilter filter, const char* name,
                                  void* ptr, OIDNFormat format,
                                  size_t width, size_t height,
                                  size_t byteOffset,
                                  size_t bytePixelStride, size_t byteRowStride);

It is possible to specify either a data buffer object (`buffer` argument) with
the `oidnSetFilterImage` function, or directly a pointer to shared user-managed
data (`ptr` argument) with the `oidnSetSharedFilterImage` function.

In both cases, you must also specify the name of the image parameter to set
(`name` argument, e.g. `"color"`, `"output"`), the pixel format (`format`
argument), the width and height of the image in number of pixels (`width` and
`height` arguments), the starting offset of the image data (`byteOffset`
argument), the pixel stride (`bytePixelStride` argument) and the row stride
(`byteRowStride` argument), in number of bytes. Note that the row stride must
be an integer multiple of the pixel stride.

If the pixels and/or rows are stored contiguously (tightly packed without any
gaps), you can set `bytePixelStride` and/or `byteRowStride` to `0` to let the
library compute the actual strides automatically, as a convenience.

Filters may have parameters other than buffers as well, which you can set and
get using the following functions:

    void oidnSetFilter1b(OIDNFilter filter, const char* name, bool value);
    void oidnSetFilter1i(OIDNFilter filter, const char* name, int  value);
    bool oidnGetFilter1b(OIDNFilter filter, const char* name);
    int  oidnGetFilter1i(OIDNFilter filter, const char* name);

After setting all necessary parameters for the filter, the changes must be
commmitted by calling

    void oidnCommitFilter(OIDNFilter filter);

The parameters can be updated after committing the filter, but it must be
re-committed for the changes to take effect.

Finally, an image can be filtered by executing the filter with

    void oidnExecuteFilter(OIDNFilter filter);

which will read the input image data from the specified buffers and produce the
denoised output image.

In the following we describe the different filters that are currently
implemented in Open Image Denoise.

### RT

The `RT` (**r**ay **t**racing) filter is a generic ray tracing denoising filter
which is suitable for denoising images rendered with Monte Carlo ray tracing
methods like path tracing. The `RT` filter is based on a deep learning based
denoising algorithm, and it aims to provide a good balance between denoising
performance and quality.

It accepts either a low dynamic range (LDR) or high dynamic
range (HDR) color image as input. Optionally, it also accepts auxiliary feature
images, e.g. albedo and normal, which improve the denoising quality, preserving
more details in the image.

The filter can be created by passing `"RT"` to the `oidnNewFilter` function
as the filter type. The filter supports the following parameters:

------- -------- ----------- -------- -----------------------------------------
Type    Format   Name         Default Description
------- -------- ----------- -------- -----------------------------------------
Image   float3   color                input color image (LDR or HDR)

Image   float3   albedo               input image containing the albedo value of
                                      the first hit per pixel; *optional*

Image   float3   normal               input image containing the normal
                                      (world-space or view-space, arbitrary
                                      length) of the first hit per pixel;
                                      *optional*, requires setting the albedo
                                      image too

Image   float3   output               output image; can be one of the input
                                      images

bool             hdr            false whether the color is HDR

bool             srgb           false whether the color is encoded with the
                                      sRGB (2.2 gamma) curve (LDR only) or is
                                      linear; the output will be encoded with
                                      the same curve
------- -------- ----------- -------- -----------------------------------------
: Parameters supported by the `RT` filter.

All specified images must have the same dimensions.

