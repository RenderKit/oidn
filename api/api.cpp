// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Initializes and locks the global context
// Use *only* inside OIDN_TRY/CATCH!
#define OIDN_INIT_CONTEXT(ctx, deviceType) \
  Context& ctx = Context::get(); \
  std::lock_guard<std::mutex> ctxGuard(ctx.getMutex()); \
  ctx.init(deviceType);

// Locks the device that owns the specified object and saves/restores state
// Use *only* inside OIDN_TRY/CATCH!
#define OIDN_LOCK_DEVICE(obj) \
  DeviceGuard deviceGuard(obj);

// Try/catch for converting exceptions to errors
#define OIDN_TRY \
  try {

#define OIDN_CATCH_DEVICE(obj) \
  } catch (const Exception& e) {                                                  \
    Device::setError(getDevice(obj), e.code(), e.what());                         \
  } catch (const std::bad_alloc&) {                                               \
    Device::setError(getDevice(obj), Error::OutOfMemory, "out of memory");        \
  } catch (const std::exception& e) {                                             \
    Device::setError(getDevice(obj), Error::Unknown, e.what());                   \
  } catch (...) {                                                                 \
    Device::setError(getDevice(obj), Error::Unknown, "unknown exception caught"); \
  }

#define OIDN_CATCH OIDN_CATCH_DEVICE(nullptr)

#include "common/common.h"
#include "core/context.h"
#include "core/engine.h"
#include "core/filter.h"
#include <mutex>

OIDN_NAMESPACE_USING
OIDN_API_NAMESPACE_BEGIN

  class DeviceGuard
  {
  public:
    template<typename T>
    DeviceGuard(T* obj)
      : device(obj->getDevice()),
        lock(device->getMutex())
    {
      device->enter(); // save state
    }

    ~DeviceGuard()
    {
      try
      {
        device->leave(); // restore state
      }
      catch (...) {}
    }

  private:
    // Disable copying
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator =(const DeviceGuard&) = delete;

    Ref<Device> device;               // ref needed to keep the device alive
    std::lock_guard<std::mutex> lock; // must be declared *after* the device
  };

  namespace
  {
    oidn_inline void checkHandle(void* handle)
    {
      if (handle == nullptr)
        throw Exception(Error::InvalidArgument, "handle is null");
    }

    oidn_inline void checkString(const char* str)
    {
      if (str == nullptr)
        throw Exception(Error::InvalidArgument, "string pointer is null");
    }

    template<typename T>
    oidn_inline Device* getDevice(T* obj)
    {
      return obj ? obj->getDevice() : nullptr;
    }

    oidn_inline Device* getDevice(std::nullptr_t)
    {
      return nullptr;
    }

    template<typename T>
    oidn_inline void retainObject(T* obj)
    {
      if (obj)
      {
        obj->incRef();
      }
      else
      {
        OIDN_TRY
          checkHandle(obj);
        OIDN_CATCH_DEVICE(obj)
      }
    }

    template<typename T>
    oidn_inline void releaseObject(T* obj)
    {
      if (obj == nullptr || obj->decRefKeep() == 0)
      {
        OIDN_TRY
          checkHandle(obj);
          OIDN_LOCK_DEVICE(obj);
          obj->getDevice()->wait(); // wait for all async operations to complete
          obj->destroy();
          obj = nullptr;
        OIDN_CATCH_DEVICE(obj)
      }
    }

    template<>
    oidn_inline void releaseObject(Device* device)
    {
      if (device == nullptr || device->decRefKeep() == 0)
      {
        OIDN_TRY
          checkHandle(device);
          // No need to lock the device because we're just destroying it
          device->enter(); // save state
          device->wait();  // wait for all async operations to complete
          device->leave(); // restore state
          device->destroy();
          device = nullptr;
        OIDN_CATCH
      }
    }
  }

  // -----------------------------------------------------------------------------------------------
  // Physical Device
  // -----------------------------------------------------------------------------------------------

  OIDN_API int oidnGetNumPhysicalDevices()
  {
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::Default);
      return ctx.getNumPhysicalDevices();
    OIDN_CATCH
    return 0;
  }

  OIDN_API bool oidnGetPhysicalDeviceBool(int physicalDeviceID, const char* name)
  {
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::Default);
      checkString(name);
      return ctx.getPhysicalDevice(physicalDeviceID)->getInt(name);
    OIDN_CATCH
    return 0;
  }

  OIDN_API int oidnGetPhysicalDeviceInt(int physicalDeviceID, const char* name)
  {
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::Default);
      checkString(name);
      return ctx.getPhysicalDevice(physicalDeviceID)->getInt(name);
    OIDN_CATCH
    return 0;
  }

  OIDN_API const char* oidnGetPhysicalDeviceString(int physicalDeviceID, const char* name)
  {
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::Default);
      checkString(name);
      return ctx.getPhysicalDevice(physicalDeviceID)->getString(name);
    OIDN_CATCH
    return nullptr;
  }

  OIDN_API const void* oidnGetPhysicalDeviceData(int physicalDeviceID, const char* name, size_t* byteSize)
  {
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::Default);
      checkString(name);
      Data data = ctx.getPhysicalDevice(physicalDeviceID)->getData(name);
      if (byteSize != nullptr)
        *byteSize = data.size;
      return data.ptr;
    OIDN_CATCH
    return nullptr;
  }

  // -----------------------------------------------------------------------------------------------
  // Device
  // -----------------------------------------------------------------------------------------------

  OIDN_API bool oidnIsCPUDeviceSupported()
  {
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::CPU);
      return ctx.isDeviceSupported(DeviceType::CPU);
    OIDN_CATCH
    return false;
  }

  OIDN_API bool oidnIsSYCLDeviceSupported(const sycl::device* device)
  {
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::SYCL);
      if (!ctx.isDeviceSupported(DeviceType::SYCL))
        return false;
      auto factory = static_cast<SYCLDeviceFactoryBase*>(ctx.getDeviceFactory(DeviceType::SYCL));
      return factory->isDeviceSupported(device);
    OIDN_CATCH
    return false;
  }

  OIDN_API bool oidnIsCUDADeviceSupported(int deviceID)
  {
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::CUDA);
      if (!ctx.isDeviceSupported(DeviceType::CUDA))
        return false;
      auto factory = static_cast<CUDADeviceFactoryBase*>(ctx.getDeviceFactory(DeviceType::CUDA));
      return factory->isDeviceSupported(deviceID);
    OIDN_CATCH
    return false;
  }

  OIDN_API bool oidnIsHIPDeviceSupported(int deviceID)
  {
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::HIP);
      if (!ctx.isDeviceSupported(DeviceType::HIP))
        return false;
      auto factory = static_cast<HIPDeviceFactoryBase*>(ctx.getDeviceFactory(DeviceType::HIP));
      return factory->isDeviceSupported(deviceID);
    OIDN_CATCH
    return false;
  }

  OIDN_API bool oidnIsMetalDeviceSupported(MTLDevice_id device)
  {
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::Metal);
      if (!ctx.isDeviceSupported(DeviceType::Metal))
        return false;
      auto factory = static_cast<MetalDeviceFactoryBase*>(ctx.getDeviceFactory(DeviceType::Metal));
      return factory->isDeviceSupported(device);
    OIDN_CATCH
    return false;
  }

  OIDN_API OIDNDevice oidnNewDevice(OIDNDeviceType inType)
  {
    DeviceType type = static_cast<DeviceType>(inType);
    Ref<Device> device = nullptr;

    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, type);

      if (type == DeviceType::Default)
      {
        const int numDevices = ctx.getNumPhysicalDevices();
        if (numDevices == 0)
          throw Exception(Error::UnsupportedHardware, "no supported devices found");

        // Check whether the user wants to override the default device
        std::string deviceStr;
        if (getEnvVar("OIDN_DEFAULT_DEVICE", deviceStr) && !deviceStr.empty())
        {
          if (isdigit(deviceStr[0]))
          {
            const int id = fromString<int>(deviceStr);
            if (id >= 0 && id < numDevices)
              device = ctx.newDevice(id);
          }
          else
          {
            try
            {
              type = fromString<DeviceType>(deviceStr);
            }
            catch (...) {}

            if (ctx.isDeviceSupported(type))
              device = ctx.newDevice(type);
          }
        }

        if (!device)
          device = ctx.newDevice(0);
      }
      else
      {
        device = ctx.newDevice(type);
      }
    OIDN_CATCH

    return reinterpret_cast<OIDNDevice>(device.detach());
  }

  OIDN_API OIDNDevice oidnNewDeviceByID(int physicalDeviceID)
  {
    Ref<Device> device = nullptr;
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::Default);
      device = ctx.newDevice(physicalDeviceID);
    OIDN_CATCH
    return reinterpret_cast<OIDNDevice>(device.detach());
  }

  OIDN_API OIDNDevice oidnNewDeviceByUUID(const void* uuid)
  {
    Ref<Device> device = nullptr;

    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::Default);

      if (uuid == nullptr)
        throw Exception(Error::InvalidArgument, "UUID pointer is null");

      // Find the physical device with the specified UUID
      const int numDevices = ctx.getNumPhysicalDevices();
      int foundID = -1;

      for (int i = 0; i < numDevices; ++i)
      {
        const auto& physicalDevice = ctx.getPhysicalDevice(i);
        if (physicalDevice->uuidSupported &&
            memcmp(uuid, physicalDevice->uuid.bytes, sizeof(physicalDevice->uuid.bytes)) == 0)
        {
          foundID = i;
          break;
        }
      }

      if (foundID < 0)
        throw Exception(Error::InvalidArgument, "no physical device found with specified UUID");

      device = ctx.newDevice(foundID);
    OIDN_CATCH

    return reinterpret_cast<OIDNDevice>(device.detach());
  }

  OIDN_API OIDNDevice oidnNewDeviceByLUID(const void* luid)
  {
    Ref<Device> device = nullptr;

    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::Default);

      if (luid == nullptr)
        throw Exception(Error::InvalidArgument, "LUID pointer is null");

      // Find the physical device with the specified LUID
      const int numDevices = ctx.getNumPhysicalDevices();
      int foundID = -1;

      for (int i = 0; i < numDevices; ++i)
      {
        const auto& physicalDevice = ctx.getPhysicalDevice(i);
        if (physicalDevice->luidSupported &&
            memcmp(luid, physicalDevice->luid.bytes, sizeof(physicalDevice->luid.bytes)) == 0)
        {
          foundID = i;
          break;
        }
      }

      if (foundID < 0)
        throw Exception(Error::InvalidArgument, "no physical device found with specified LUID");

      device = ctx.newDevice(foundID);
    OIDN_CATCH

    return reinterpret_cast<OIDNDevice>(device.detach());
  }

  OIDN_API OIDNDevice oidnNewDeviceByPCIAddress(int pciDomain, int pciBus, int pciDevice, int pciFunction)
  {
    Ref<Device> device = nullptr;

    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::Default);

      // Find the physical device with the specified PCI address
      const int numDevices = ctx.getNumPhysicalDevices();
      int foundID = -1;

      for (int i = 0; i < numDevices; ++i)
      {
        const auto& physicalDevice = ctx.getPhysicalDevice(i);
        if (physicalDevice->pciAddressSupported &&
            physicalDevice->pciDomain   == pciDomain &&
            physicalDevice->pciBus      == pciBus    &&
            physicalDevice->pciDevice   == pciDevice &&
            physicalDevice->pciFunction == pciFunction)
        {
          foundID = i;
          break;
        }
      }

      if (foundID < 0)
        throw Exception(Error::InvalidArgument, "no physical device found with specified PCI address");

      device = ctx.newDevice(foundID);
    OIDN_CATCH

    return reinterpret_cast<OIDNDevice>(device.detach());
  }

  OIDN_API OIDNDevice oidnNewSYCLDevice(const sycl::queue* queues, int numQueues)
  {
    Ref<Device> device = nullptr;
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::SYCL);
      auto factory = static_cast<SYCLDeviceFactoryBase*>(ctx.getDeviceFactory(DeviceType::SYCL));
      device = factory->newDevice(queues, numQueues);
    OIDN_CATCH
    return reinterpret_cast<OIDNDevice>(device.detach());
  }

  OIDN_API OIDNDevice oidnNewCUDADevice(const int* deviceIDs, const cudaStream_t* streams, int numPairs)
  {
    Ref<Device> device = nullptr;
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::CUDA);
      auto factory = static_cast<CUDADeviceFactoryBase*>(ctx.getDeviceFactory(DeviceType::CUDA));
      device = factory->newDevice(deviceIDs, streams, numPairs);
    OIDN_CATCH
    return reinterpret_cast<OIDNDevice>(device.detach());
  }

  OIDN_API OIDNDevice oidnNewHIPDevice(const int* deviceIDs, const hipStream_t* streams, int numPairs)
  {
    Ref<Device> device = nullptr;
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::HIP);
      auto factory = static_cast<HIPDeviceFactoryBase*>(ctx.getDeviceFactory(DeviceType::HIP));
      device = factory->newDevice(deviceIDs, streams, numPairs);
    OIDN_CATCH
    return reinterpret_cast<OIDNDevice>(device.detach());
  }

  OIDN_API OIDNDevice oidnNewMetalDevice(const MTLCommandQueue_id* commandQueues, int numQueues)
  {
    Ref<Device> device = nullptr;
    OIDN_TRY
      OIDN_INIT_CONTEXT(ctx, DeviceType::Metal);
      auto factory = static_cast<MetalDeviceFactoryBase*>(ctx.getDeviceFactory(DeviceType::Metal));
      device = factory->newDevice(commandQueues, numQueues);
    OIDN_CATCH
    return reinterpret_cast<OIDNDevice>(device.detach());
  }

  OIDN_API void oidnRetainDevice(OIDNDevice deviceC)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    retainObject(device);
  }

  OIDN_API void oidnReleaseDevice(OIDNDevice deviceC)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    releaseObject(device);
  }

  OIDN_API void oidnSetDeviceBool(OIDNDevice deviceC, const char* name, bool value)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      checkString(name);
      device->setInt(name, value);
    OIDN_CATCH_DEVICE(device)
  }

  OIDN_API void oidnSetDeviceInt(OIDNDevice deviceC, const char* name, int value)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      checkString(name);
      device->setInt(name, value);
    OIDN_CATCH_DEVICE(device)
  }

  OIDN_API bool oidnGetDeviceBool(OIDNDevice deviceC, const char* name)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      checkString(name);
      return device->getInt(name);
    OIDN_CATCH_DEVICE(device)
    return false;
  }

  OIDN_API int oidnGetDeviceInt(OIDNDevice deviceC, const char* name)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      checkString(name);
      return device->getInt(name);
    OIDN_CATCH_DEVICE(device)
    return 0;
  }

  OIDN_API void oidnSetDeviceErrorFunction(OIDNDevice deviceC, OIDNErrorFunction func, void* userPtr)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->setErrorFunction(reinterpret_cast<ErrorFunction>(func), userPtr);
    OIDN_CATCH_DEVICE(device)
  }

  OIDN_API OIDNError oidnGetDeviceError(OIDNDevice deviceC, const char** outMessage)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      return static_cast<OIDNError>(Device::getError(device, outMessage));
    OIDN_CATCH_DEVICE(device)
    if (outMessage) *outMessage = "";
    return OIDN_ERROR_UNKNOWN;
  }

  OIDN_API void oidnCommitDevice(OIDNDevice deviceC)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->commit();
    OIDN_CATCH_DEVICE(device)
  }

  OIDN_API void oidnSyncDevice(OIDNDevice deviceC)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->checkCommitted();
      device->waitAndThrow();
    OIDN_CATCH_DEVICE(device)
  }

  // -----------------------------------------------------------------------------------------------
  // Buffer
  // -----------------------------------------------------------------------------------------------

  OIDN_API OIDNBuffer oidnNewBuffer(OIDNDevice deviceC, size_t byteSize)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->checkCommitted();
      Ref<Buffer> buffer = device->newUserBuffer(byteSize, Storage::Undefined);
      return reinterpret_cast<OIDNBuffer>(buffer.detach());
    OIDN_CATCH_DEVICE(device)
    return nullptr;
  }

  OIDN_API OIDNBuffer oidnNewBufferWithStorage(OIDNDevice deviceC, size_t byteSize, OIDNStorage storage)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->checkCommitted();
      Ref<Buffer> buffer = device->newUserBuffer(byteSize, static_cast<Storage>(storage));
      return reinterpret_cast<OIDNBuffer>(buffer.detach());
    OIDN_CATCH_DEVICE(device)
    return nullptr;
  }

  OIDN_API OIDNBuffer oidnNewSharedBuffer(OIDNDevice deviceC, void* devPtr, size_t byteSize)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->checkCommitted();
      Ref<Buffer> buffer = device->newUserBuffer(devPtr, byteSize);
      return reinterpret_cast<OIDNBuffer>(buffer.detach());
    OIDN_CATCH_DEVICE(device)
    return nullptr;
  }

  OIDN_API OIDNBuffer oidnNewSharedBufferFromFD(OIDNDevice deviceC,
                                                OIDNExternalMemoryTypeFlags fdTypeC,
                                                int fd, size_t byteSize)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->checkCommitted();
      const ExternalMemoryTypeFlags fdType = static_cast<ExternalMemoryTypeFlags>(fdTypeC);
      if ((fdType & device->getExternalMemoryTypes()) != fdType)
        throw Exception(Error::InvalidArgument, "external memory type not supported by the device");
      Ref<Buffer> buffer = device->newExternalUserBuffer(fdType, fd, byteSize);
      return reinterpret_cast<OIDNBuffer>(buffer.detach());
    OIDN_CATCH_DEVICE(device)
    return nullptr;
  }

  OIDN_API OIDNBuffer oidnNewSharedBufferFromWin32Handle(OIDNDevice deviceC,
                                                         OIDNExternalMemoryTypeFlags handleTypeC,
                                                         void* handle, const void* name, size_t byteSize)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->checkCommitted();
      const ExternalMemoryTypeFlags handleType = static_cast<ExternalMemoryTypeFlags>(handleTypeC);
      if ((handleType & device->getExternalMemoryTypes()) != handleType)
        throw Exception(Error::InvalidArgument, "external memory type not supported by the device");
      if ((!handle && !name) || (handle && name))
        throw Exception(Error::InvalidArgument, "exactly one of the external memory handle and name must be non-null");
      Ref<Buffer> buffer = device->newExternalUserBuffer(handleType, handle, name, byteSize);
      return reinterpret_cast<OIDNBuffer>(buffer.detach());
    OIDN_CATCH_DEVICE(device)
    return nullptr;
  }

  OIDN_API OIDNBuffer oidnNewSharedBufferFromMetal(OIDNDevice deviceC, MTLBuffer_id mtlBuffer)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->checkCommitted();
      Ref<Buffer> buffer = device->newNativeUserBuffer(mtlBuffer);
      return reinterpret_cast<OIDNBuffer>(buffer.detach());
    OIDN_CATCH_DEVICE(device)
    return nullptr;
  }

  OIDN_API void oidnRetainBuffer(OIDNBuffer bufferC)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(bufferC);
    retainObject(buffer);
  }

  OIDN_API void oidnReleaseBuffer(OIDNBuffer bufferC)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(bufferC);
    releaseObject(buffer);
  }

  OIDN_API void oidnReadBuffer(OIDNBuffer bufferC, size_t byteOffset, size_t byteSize, void* dstHostPtr)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(bufferC);
    OIDN_TRY
      checkHandle(bufferC);
      OIDN_LOCK_DEVICE(buffer);
      buffer->read(byteOffset, byteSize, dstHostPtr);
    OIDN_CATCH_DEVICE(buffer);
  }

  OIDN_API void oidnReadBufferAsync(OIDNBuffer bufferC,
                                    size_t byteOffset, size_t byteSize, void* dstHostPtr)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(bufferC);
    OIDN_TRY
      checkHandle(bufferC);
      OIDN_LOCK_DEVICE(buffer);
      buffer->read(byteOffset, byteSize, dstHostPtr, SyncMode::Async);
    OIDN_CATCH_DEVICE(buffer);
  }

  OIDN_API void oidnWriteBuffer(OIDNBuffer bufferC,
                                size_t byteOffset, size_t byteSize, const void* srcHostPtr)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(bufferC);
    OIDN_TRY
      checkHandle(bufferC);
      OIDN_LOCK_DEVICE(buffer);
      buffer->write(byteOffset, byteSize, srcHostPtr);
    OIDN_CATCH_DEVICE(buffer);
  }

  OIDN_API void oidnWriteBufferAsync(OIDNBuffer bufferC,
                                     size_t byteOffset, size_t byteSize, const void* srcHostPtr)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(bufferC);
    OIDN_TRY
      checkHandle(bufferC);
      OIDN_LOCK_DEVICE(buffer);
      buffer->write(byteOffset, byteSize, srcHostPtr, SyncMode::Async);
    OIDN_CATCH_DEVICE(buffer);
  }

  OIDN_API size_t oidnGetBufferSize(OIDNBuffer bufferC)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(bufferC);
    OIDN_TRY
      checkHandle(bufferC);
      OIDN_LOCK_DEVICE(buffer);
      return buffer->getByteSize();
    OIDN_CATCH_DEVICE(buffer)
    return 0;
  }

  OIDN_API OIDNStorage oidnGetBufferStorage(OIDNBuffer bufferC)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(bufferC);
    OIDN_TRY
      checkHandle(bufferC);
      OIDN_LOCK_DEVICE(buffer);
      return static_cast<OIDNStorage>(buffer->getStorage());
    OIDN_CATCH_DEVICE(buffer)
    return OIDN_STORAGE_UNDEFINED;
  }

  OIDN_API void* oidnGetBufferData(OIDNBuffer bufferC)
  {
    Buffer* buffer = reinterpret_cast<Buffer*>(bufferC);
    OIDN_TRY
      checkHandle(bufferC);
      OIDN_LOCK_DEVICE(buffer);
      if (void* ptr = buffer->getHostPtr())
        return ptr;
      else
        return buffer->getPtr();
    OIDN_CATCH_DEVICE(buffer)
    return nullptr;
  }

  // -----------------------------------------------------------------------------------------------
  // Semaphore
  // -----------------------------------------------------------------------------------------------

  OIDN_API OIDNSemaphore oidnNewSharedSemaphoreFromFD(OIDNDevice deviceC,
                                                      OIDNExternalSemaphoreTypeFlags fdTypeC,
                                                      int fd)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->checkCommitted();
      const ExternalSemaphoreTypeFlags fdType = static_cast<ExternalSemaphoreTypeFlag>(fdTypeC);
      if ((fdType & device->getExternalSemaphoreTypes()) != fdType)
        throw Exception(Error::InvalidArgument, "external semaphore type not supported by the device");
      Ref<Semaphore> semaphore = device->newExternalSemaphore(fdType, fd);
      return reinterpret_cast<OIDNSemaphore>(semaphore.detach());
    OIDN_CATCH_DEVICE(device)
    return nullptr;
  }

  OIDN_API OIDNSemaphore oidnNewSharedSemaphoreFromWin32Handle(OIDNDevice deviceC,
                                                               OIDNExternalSemaphoreTypeFlags handleTypeC,
                                                               void* handle, const void* name)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->checkCommitted();
      const ExternalSemaphoreTypeFlags handleType = static_cast<ExternalSemaphoreTypeFlag>(handleTypeC);
      if ((handleType & device->getExternalSemaphoreTypes()) != handleType)
        throw Exception(Error::InvalidArgument, "external semaphore type not supported by the device");
      if ((!handle && !name) || (handle && name))
        throw Exception(Error::InvalidArgument, "exactly one of the external semaphore handle and name must be non-null");
      Ref<Semaphore> semaphore = device->newExternalSemaphore(handleType, handle, name);
      return reinterpret_cast<OIDNSemaphore>(semaphore.detach());
    OIDN_CATCH_DEVICE(device)
    return nullptr;
  }

  OIDN_API void oidnSignalSemaphoresAsync(OIDNDevice deviceC,
                                          const OIDNSemaphore* semaphoresC,
                                          const uint64_t* values,
                                          int numSemaphores)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->checkCommitted();
      device->submitSignalSemaphores(
        reinterpret_cast<Semaphore* const*>(semaphoresC),
        values,
        numSemaphores);
    OIDN_CATCH_DEVICE(device)
  }

  OIDN_API void oidnWaitSemaphoresAsync(OIDNDevice deviceC,
                                        const OIDNSemaphore* semaphoresC,
                                        const uint64_t* values,
                                        const uint32_t* timeoutsMs,
                                        int numSemaphores)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->checkCommitted();
      device->submitWaitSemaphores(
        reinterpret_cast<Semaphore* const*>(semaphoresC),
        values,
        timeoutsMs,
        numSemaphores);
    OIDN_CATCH_DEVICE(device)
  }

  OIDN_API void oidnRetainSemaphore(OIDNSemaphore semaphoreC)
  {
    Semaphore* semaphore = reinterpret_cast<Semaphore*>(semaphoreC);
    retainObject(semaphore);
  }

  OIDN_API void oidnReleaseSemaphore(OIDNSemaphore semaphoreC)
  {
    Semaphore* semaphore = reinterpret_cast<Semaphore*>(semaphoreC);
    releaseObject(semaphore);
  }

  // -----------------------------------------------------------------------------------------------
  // Filter
  // -----------------------------------------------------------------------------------------------

  OIDN_API OIDNFilter oidnNewFilter(OIDNDevice deviceC, const char* type)
  {
    Device* device = reinterpret_cast<Device*>(deviceC);
    OIDN_TRY
      checkHandle(deviceC);
      OIDN_LOCK_DEVICE(device);
      device->checkCommitted();
      checkString(type);
      Ref<Filter> filter = device->newFilter(type);
      return reinterpret_cast<OIDNFilter>(filter.detach());
    OIDN_CATCH_DEVICE(device)
    return nullptr;
  }

  OIDN_API void oidnRetainFilter(OIDNFilter filterC)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    retainObject(filter);
  }

  OIDN_API void oidnReleaseFilter(OIDNFilter filterC)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    releaseObject(filter);
  }

  OIDN_API void oidnSetFilterImage(OIDNFilter filterC, const char* name,
                                   OIDNBuffer bufferC, OIDNFormat format,
                                   size_t width, size_t height,
                                   size_t byteOffset,
                                   size_t pixelByteStride, size_t rowByteStride)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      checkString(name);
      checkHandle(bufferC);
      Ref<Buffer> buffer = reinterpret_cast<Buffer*>(bufferC);
      if (buffer->getDevice() != filter->getDevice())
        throw Exception(Error::InvalidArgument, "the specified objects are bound to different devices");
      auto image = makeRef<Image>(buffer, static_cast<Format>(format),
                                  static_cast<int>(width), static_cast<int>(height),
                                  byteOffset, pixelByteStride, rowByteStride);
      filter->setImage(name, image);
    OIDN_CATCH_DEVICE(filter)
  }

  OIDN_API void oidnSetSharedFilterImage(OIDNFilter filterC, const char* name,
                                         void* devPtr, OIDNFormat format,
                                         size_t width, size_t height,
                                         size_t byteOffset,
                                         size_t pixelByteStride, size_t rowByteStride)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      checkString(name);
      auto image = makeRef<Image>(devPtr, static_cast<Format>(format),
                                  static_cast<int>(width), static_cast<int>(height),
                                  byteOffset, pixelByteStride, rowByteStride);
      filter->setImage(name, image);
    OIDN_CATCH_DEVICE(filter)
  }

  OIDN_API void oidnUnsetFilterImage(OIDNFilter filterC, const char* name)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      checkString(name);
      filter->unsetImage(name);
    OIDN_CATCH_DEVICE(filter)
  }

  OIDN_API void oidnSetSharedFilterData(OIDNFilter filterC, const char* name,
                                        void* hostPtr, size_t byteSize)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      checkString(name);
      Data data(hostPtr, byteSize);
      filter->setData(name, data);
    OIDN_CATCH_DEVICE(filter)
  }

  OIDN_API void oidnUpdateFilterData(OIDNFilter filterC, const char* name)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      checkString(name);
      filter->updateData(name);
    OIDN_CATCH_DEVICE(filter)
  }

  OIDN_API void oidnUnsetFilterData(OIDNFilter filterC, const char* name)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      checkString(name);
      filter->unsetData(name);
    OIDN_CATCH_DEVICE(filter)
  }

  OIDN_API void oidnSetFilterBool(OIDNFilter filterC, const char* name, bool value)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      checkString(name);
      filter->setInt(name, int(value));
    OIDN_CATCH_DEVICE(filter)
  }

  OIDN_API bool oidnGetFilterBool(OIDNFilter filterC, const char* name)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      checkString(name);
      return filter->getInt(name);
    OIDN_CATCH_DEVICE(filter)
    return false;
  }

  OIDN_API void oidnSetFilterInt(OIDNFilter filterC, const char* name, int value)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      checkString(name);
      filter->setInt(name, value);
    OIDN_CATCH_DEVICE(filter)
  }

  OIDN_API int oidnGetFilterInt(OIDNFilter filterC, const char* name)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      checkString(name);
      return filter->getInt(name);
    OIDN_CATCH_DEVICE(filter)
    return 0;
  }

  OIDN_API void oidnSetFilterFloat(OIDNFilter filterC, const char* name, float value)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      checkString(name);
      filter->setFloat(name, value);
    OIDN_CATCH_DEVICE(filter)
  }

  OIDN_API float oidnGetFilterFloat(OIDNFilter filterC, const char* name)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      checkString(name);
      return filter->getFloat(name);
    OIDN_CATCH_DEVICE(filter)
    return 0;
  }

  OIDN_API void oidnSetFilterProgressMonitorFunction(OIDNFilter filterC,
                                                     OIDNProgressMonitorFunction func, void* userPtr)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      filter->setProgressMonitorFunction(func, userPtr);
    OIDN_CATCH_DEVICE(filter)
  }

  OIDN_API void oidnCommitFilter(OIDNFilter filterC)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      filter->commit();
    OIDN_CATCH_DEVICE(filter)
  }

  OIDN_API void oidnExecuteFilter(OIDNFilter filterC)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      filter->execute();
    OIDN_CATCH_DEVICE(filter)
  }

  OIDN_API void oidnExecuteFilterAsync(OIDNFilter filterC)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);
      filter->execute(SyncMode::Async);
    OIDN_CATCH_DEVICE(filter)
  }

  OIDN_API void oidnExecuteSYCLFilterAsync(OIDNFilter filterC,
                                           const sycl::event* depEvents, int numDepEvents,
                                           sycl::event* doneEvent)
  {
    Filter* filter = reinterpret_cast<Filter*>(filterC);
    OIDN_TRY
      checkHandle(filterC);
      OIDN_LOCK_DEVICE(filter);

      // Check whether the filter belongs to a SYCL device
      if (filter->getDevice()->getType() != DeviceType::SYCL)
        throw Exception(Error::InvalidOperation, "filter does not belong to a SYCL device");
      SYCLDeviceBase* device = static_cast<SYCLDeviceBase*>(filter->getDevice());

      // Execute the filter
      device->setDepEvents(depEvents, numDepEvents);
      filter->execute(SyncMode::Async);

      // Output the completion event (optional)
      if (doneEvent != nullptr)
        device->getDoneEvent(*doneEvent);
    OIDN_CATCH_DEVICE(filter)
  }

OIDN_API_NAMESPACE_END
