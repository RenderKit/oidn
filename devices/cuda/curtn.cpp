// CURTN: a nano implementation of the CUDA Runtime API on top of the Driver API
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "curtn.h"
#include <fatbinary_section.h>
#include <memory>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <mutex>

#if CUDA_VERSION < 11080
  #error CUDA version 11.8 or higher is required for compilation
#endif

// CUDA driver/runtime error definitions
#if CUDA_VERSION >= 12000
  #define CURTN_DEFINE_ERROR_12_0(a, b) CURTN_DEFINE_ERROR(a, b)
#else
  #define CURTN_DEFINE_ERROR_12_0(a, b)
#endif

#if CUDA_VERSION >= 12010
  #define CURTN_DEFINE_ERROR_12_1(a, b) CURTN_DEFINE_ERROR(a, b)
#else
  #define CURTN_DEFINE_ERROR_12_1(a, b)
#endif

#define CURTN_DEFINE_ERRORS \
  CURTN_DEFINE_ERROR(CUDA_SUCCESS,                              cudaSuccess) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_INVALID_VALUE,                  cudaErrorInvalidValue) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_OUT_OF_MEMORY,                  cudaErrorMemoryAllocation) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_NOT_INITIALIZED,                cudaErrorInitializationError) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_DEINITIALIZED,                  cudaErrorCudartUnloading) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_PROFILER_DISABLED,              cudaErrorProfilerDisabled) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_STUB_LIBRARY,                   cudaErrorStubLibrary) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_DEVICE_UNAVAILABLE,             cudaErrorDevicesUnavailable) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_NO_DEVICE,                      cudaErrorNoDevice) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_INVALID_DEVICE,                 cudaErrorInvalidDevice) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_DEVICE_NOT_LICENSED,            cudaErrorDeviceNotLicensed) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_INVALID_IMAGE,                  cudaErrorInvalidKernelImage) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_INVALID_CONTEXT,                cudaErrorIncompatibleDriverContext) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_MAP_FAILED,                     cudaErrorMapBufferObjectFailed) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_UNMAP_FAILED,                   cudaErrorUnmapBufferObjectFailed) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_ARRAY_IS_MAPPED,                cudaErrorArrayIsMapped) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_ALREADY_MAPPED,                 cudaErrorAlreadyMapped) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_NO_BINARY_FOR_GPU,              cudaErrorNoKernelImageForDevice) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_ALREADY_ACQUIRED,               cudaErrorAlreadyAcquired) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_NOT_MAPPED,                     cudaErrorNotMapped) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_NOT_MAPPED_AS_ARRAY,            cudaErrorNotMappedAsArray) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_NOT_MAPPED_AS_POINTER,          cudaErrorNotMappedAsPointer) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_ECC_UNCORRECTABLE,              cudaErrorECCUncorrectable) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_UNSUPPORTED_LIMIT,              cudaErrorUnsupportedLimit) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_CONTEXT_ALREADY_IN_USE,         cudaErrorDeviceAlreadyInUse) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,        cudaErrorPeerAccessUnsupported) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_INVALID_PTX,                    cudaErrorInvalidPtx) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,       cudaErrorInvalidGraphicsContext) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_NVLINK_UNCORRECTABLE,           cudaErrorNvlinkUncorrectable) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_JIT_COMPILER_NOT_FOUND,         cudaErrorJitCompilerNotFound) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_UNSUPPORTED_PTX_VERSION,        cudaErrorUnsupportedPtxVersion) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_JIT_COMPILATION_DISABLED,       cudaErrorJitCompilationDisabled) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY,      cudaErrorUnsupportedExecAffinity) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_INVALID_SOURCE,                 cudaErrorInvalidSource) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_FILE_NOT_FOUND,                 cudaErrorFileNotFound) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND, cudaErrorSharedObjectSymbolNotFound) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,      cudaErrorSharedObjectInitFailed) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_OPERATING_SYSTEM,               cudaErrorOperatingSystem) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_INVALID_HANDLE,                 cudaErrorInvalidResourceHandle) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_ILLEGAL_STATE,                  cudaErrorIllegalState) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_NOT_FOUND,                      cudaErrorInvalidDeviceFunction) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_NOT_READY,                      cudaErrorNotReady) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_ILLEGAL_ADDRESS,                cudaErrorIllegalAddress) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,        cudaErrorLaunchOutOfResources) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_LAUNCH_TIMEOUT,                 cudaErrorLaunchTimeout) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,  cudaErrorLaunchIncompatibleTexturing) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,    cudaErrorPeerAccessAlreadyEnabled) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,        cudaErrorPeerAccessNotEnabled) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_CONTEXT_IS_DESTROYED,           cudaErrorContextIsDestroyed) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_ASSERT,                         cudaErrorAssert) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_TOO_MANY_PEERS,                 cudaErrorTooManyPeers) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED, cudaErrorHostMemoryAlreadyRegistered) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,     cudaErrorHostMemoryNotRegistered) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_HARDWARE_STACK_ERROR,           cudaErrorHardwareStackError) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_ILLEGAL_INSTRUCTION,            cudaErrorIllegalInstruction) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_MISALIGNED_ADDRESS,             cudaErrorMisalignedAddress) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_INVALID_ADDRESS_SPACE,          cudaErrorInvalidAddressSpace) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_INVALID_PC,                     cudaErrorInvalidPc) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_LAUNCH_FAILED,                  cudaErrorLaunchFailure) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,   cudaErrorCooperativeLaunchTooLarge) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_NOT_PERMITTED,                  cudaErrorNotPermitted) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_NOT_SUPPORTED,                  cudaErrorNotSupported) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_SYSTEM_NOT_READY,               cudaErrorSystemNotReady) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,         cudaErrorSystemDriverMismatch) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE, cudaErrorCompatNotSupportedOnDevice) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_MPS_CONNECTION_FAILED,          cudaErrorMpsConnectionFailed) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_MPS_RPC_FAILURE,                cudaErrorMpsRpcFailure) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_MPS_SERVER_NOT_READY,           cudaErrorMpsServerNotReady) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_MPS_MAX_CLIENTS_REACHED,        cudaErrorMpsMaxClientsReached) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED,    cudaErrorMpsMaxConnectionsReached) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_MPS_CLIENT_TERMINATED,          cudaErrorMpsClientTerminated) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,     cudaErrorStreamCaptureUnsupported) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,     cudaErrorStreamCaptureInvalidated) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_STREAM_CAPTURE_MERGE,           cudaErrorStreamCaptureMerge) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_STREAM_CAPTURE_UNMATCHED,       cudaErrorStreamCaptureUnmatched) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_STREAM_CAPTURE_UNJOINED,        cudaErrorStreamCaptureUnjoined) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_STREAM_CAPTURE_ISOLATION,       cudaErrorStreamCaptureIsolation) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_STREAM_CAPTURE_IMPLICIT,        cudaErrorStreamCaptureImplicit) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_CAPTURED_EVENT,                 cudaErrorCapturedEvent) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD,    cudaErrorStreamCaptureWrongThread) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_TIMEOUT,                        cudaErrorTimeout) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE,      cudaErrorGraphExecUpdateFailure) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_EXTERNAL_DEVICE,                cudaErrorExternalDevice) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_INVALID_CLUSTER_SIZE,           cudaErrorInvalidClusterSize) \
  CURTN_DEFINE_ERROR(CUDA_ERROR_UNKNOWN,                        cudaErrorUnknown) \
  CURTN_DEFINE_ERROR_12_0(CUDA_ERROR_CDP_NOT_SUPPORTED,         cudaErrorCdpNotSupported) \
  CURTN_DEFINE_ERROR_12_0(CUDA_ERROR_CDP_VERSION_MISMATCH,      cudaErrorCdpVersionMismatch) \
  CURTN_DEFINE_ERROR_12_1(CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC,  cudaErrorUnsupportedDevSideSync)

namespace curtn
{
  // Registered function descriptor
  struct FunctionDesc
  {
    void** fatCubinHandle;
    const char* deviceName;
  };

  // Kernel call configuration
  struct CallConfig
  {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;
  };

  // Stream callback wrapper descriptor
  struct StreamCallbackDesc
  {
    cudaStreamCallback_t callback;
    void* userData;
  };

  // Per-context state
  struct ContextState
  {
    std::unordered_map<void**, CUmodule> modules;      // module handles by fatCubinHandle
    std::unordered_map<const void*, CUfunction> funcs; // function handles by symbol
    uint64_t refCount = 0;                             // reference count
  };

  // Global runtime state
  struct Runtime
  {
    // Returns the runtime singleton, calling cuInit the first time
    static Runtime& get()
    {
      static Runtime runtime;
      return runtime;
    }

    // Converts a CUresult to a cudaError_t
    static cudaError_t toError(CUresult result)
    {
      switch (result)
      {
      #define CURTN_DEFINE_ERROR(res, err) case res: return err;
      CURTN_DEFINE_ERRORS
      #undef CURTN_DEFINE_ERROR

      default: return cudaErrorUnknown;
      }
    }

    // Sets the last error on the current thread, if it's not already set
    static cudaError_t setError(cudaError_t error)
    {
      if (lastError == cudaSuccess)
        lastError = error;

      return error;
    }

    static cudaError_t setError(CUresult result)
    {
      return setError(toError(result));
    }

    // Wraps a CUDA runtime stream callback
    static void CUDA_CB streamCallback(CUstream stream, CUresult status, void* userData)
    {
      std::unique_ptr<StreamCallbackDesc> desc(reinterpret_cast<StreamCallbackDesc*>(userData));
      cudaError_t error = Runtime::toError(status);
      desc->callback(stream, error, desc->userData);
    }

    // Prepares the current context for runtime API usage
    CUresult initContext()
    {
      // Get the current context state
      CUcontext context;
      CUresult result = cuCtxGetCurrent(&context);
      if (result != CUDA_SUCCESS)
        return result;

    #if CUDA_VERSION >= 12000
      unsigned long long contextID;
      result = cuCtxGetId(context, &contextID);
      if (result != CUDA_SUCCESS)
        return result;

      std::lock_guard<std::mutex> lock(mutex);
      ContextState& cs = contextStates[contextID];
    #else
      std::lock_guard<std::mutex> lock(mutex);
      ContextState& cs = contextStates[context];
    #endif

      // Increment the reference count
      ++cs.refCount;
      return CUDA_SUCCESS;
    }

    // Releases all runtime resources associated with the current context
    CUresult cleanupContext()
    {
      // Get the current context state
      CUcontext context;
      CUresult result = cuCtxGetCurrent(&context);
      if (result != CUDA_SUCCESS)
        return result;

    #if CUDA_VERSION >= 12000
      unsigned long long contextID;
      result = cuCtxGetId(context, &contextID);
      if (result != CUDA_SUCCESS)
        return result;

      std::lock_guard<std::mutex> lock(mutex);
      ContextState& cs = contextStates[contextID];
    #else
      std::lock_guard<std::mutex> lock(mutex);
      ContextState& cs = contextStates[context];
    #endif

      // Decrement the reference count
      if (--cs.refCount > 0)
        return CUDA_SUCCESS; // context is still in use

      // Unload all modules from the context
      for (const auto& moduleItem : cs.modules)
      {
        const CUresult unloadResult = cuModuleUnload(moduleItem.second);
        if (result == CUDA_SUCCESS)
          result = unloadResult;
      }

      // Delete the context state
    #if CUDA_VERSION >= 12000
      contextStates.erase(contextID);
    #else
      contextStates.erase(context);
    #endif
      return result;
    }

    // Returns the function handle for the given symbol
    CUresult getFunction(const void* funcSymbol, CUfunction& func)
    {
      // Get the current context state
      CUcontext context;
      CUresult result = cuCtxGetCurrent(&context);
      if (result != CUDA_SUCCESS)
        return result;

    #if CUDA_VERSION >= 12000
      unsigned long long contextID;
      result = cuCtxGetId(context, &contextID);
      if (result != CUDA_SUCCESS)
        return result;

      std::lock_guard<std::mutex> lock(mutex);
      ContextState& cs = contextStates[contextID];
    #else
      std::lock_guard<std::mutex> lock(mutex);
      ContextState& cs = contextStates[context];
    #endif

      // Look up the function handle in the context state
      auto funcIter = cs.funcs.find(funcSymbol);
      if (funcIter == cs.funcs.end())
      {
        // The function handle has not been cached yet, look up the module handle
        const FunctionDesc& funcDesc = funcDescs[funcSymbol];
        CUmodule module;

        auto moduleIter = cs.modules.find(funcDesc.fatCubinHandle);
        if (moduleIter == cs.modules.end())
        {
          // The module handle has not been cached yet, load the module
          result = cuModuleLoadFatBinary(&module, *funcDesc.fatCubinHandle);
          if (result != CUDA_SUCCESS)
            return result;

          cs.modules[funcDesc.fatCubinHandle] = module;
        }
        else
          module = moduleIter->second;

        // Get the function handle
        result = cuModuleGetFunction(&func, module, funcDesc.deviceName);
        if (result != CUDA_SUCCESS)
          return result;

        cs.funcs[funcSymbol] = func;
      }
      else
        func = funcIter->second;

      return CUDA_SUCCESS;
    }

    static thread_local std::vector<CallConfig> callConfigs;
    static thread_local cudaError_t lastError;

  #if CUDA_VERSION >= 12000
    std::unordered_map<unsigned long long, ContextState> contextStates; // context states by ID
  #else
    std::unordered_map<CUcontext, ContextState> contextStates; // context states by handle
  #endif
    std::unordered_map<const void*, FunctionDesc> funcDescs; // function descriptors by symbol
    std::mutex mutex;
  };

  thread_local std::vector<CallConfig> Runtime::callConfigs;
  thread_local cudaError_t Runtime::lastError = cudaSuccess;

  cudaError_t init()
  {
    // Initialize the CUDA Driver API
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS)
      return Runtime::setError(result);

    // Check the major driver version
    int version = 0;
    result = cuDriverGetVersion(&version);
    if (result != CUDA_SUCCESS)
      return Runtime::setError(result);
    if (version < (CUDA_VERSION / 1000 * 1000))
      return Runtime::setError(cudaErrorInsufficientDriver);

    return cudaSuccess;
  }

  cudaError_t initContext()
  {
    return Runtime::setError(Runtime::get().initContext());
  }

  cudaError_t cleanupContext()
  {
    return Runtime::setError(Runtime::get().cleanupContext());
  }

  extern "C"
  {
    void** CUDARTAPI __cudaRegisterFatBinary(void* fatCubin)
    {
      __fatBinC_Wrapper_t* fatCubinWrapper = (__fatBinC_Wrapper_t*)fatCubin;
      return (void**)&fatCubinWrapper->data;
    }

    void CUDARTAPI __cudaRegisterFatBinaryEnd(void** fatCubinHandle)
    {
    }

    void CUDARTAPI __cudaUnregisterFatBinary(void** fatCubinHandle)
    {
    }

    char CUDARTAPI __cudaInitModule(void** fatCubinHandle)
    {
      return 0;
    }

    void CUDARTAPI __cudaRegisterFunction(void** fatCubinHandle,
                                          const char* hostFun,
                                          char* deviceFun,
                                          const char* deviceName,
                                          int thread_limit,
                                          uint3* tid,
                                          uint3* bid,
                                          dim3* bDim,
                                          dim3* gDim,
                                          int* wSize)
    {
      Runtime::get().funcDescs[hostFun] = {fatCubinHandle, deviceName};
    }

    void CUDARTAPI __cudaRegisterVar(void** fatCubinHandle,
                                     char* hostVar,
                                     char* deviceAddress,
                                     const char *deviceName,
                                     int ext,
                                     size_t size,
                                     int constant,
                                     int global)
    {
      // TODO: implement
    }

    unsigned int CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim,
                                                       dim3 blockDim,
                                                       size_t sharedMem,
                                                       cudaStream_t stream)
    {
      Runtime::callConfigs.push_back({gridDim, blockDim, sharedMem, stream});
      return 0;
    }

    cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3* gridDim,
                                                     dim3* blockDim,
                                                     size_t* sharedMem,
                                                     cudaStream_t* stream)
    {
      if (Runtime::callConfigs.empty())
        return Runtime::setError(cudaErrorUnknown);

      const CallConfig& config = Runtime::callConfigs.back();
      *gridDim   = config.gridDim;
      *blockDim  = config.blockDim;
      *sharedMem = config.sharedMem;
      *stream    = config.stream;

      Runtime::callConfigs.pop_back();
      return cudaSuccess;
    }

    cudaError_t CUDARTAPI cudaGetLastError()
    {
      cudaError_t error = Runtime::lastError;
      Runtime::lastError = cudaSuccess;
      return error;
    }

    const char* CUDARTAPI cudaGetErrorString(cudaError_t error)
    {
      static const char* unrecognizedStr = "unrecognized error code";

      // Convert cudaError_t to CUresult if there is an equivalent
      CUresult result;

      switch (error)
      {
      #define CURTN_DEFINE_ERROR(res, err) case err: result = res; break;
      CURTN_DEFINE_ERRORS
      #undef CURTN_DEFINE_ERROR

      case cudaErrorInvalidMemcpyDirection: return "invalid copy direction for memcpy";
      default:                              return unrecognizedStr;
      }

      // Get the error string for the CUresult
      const char* str;
      if (cuGetErrorString(result, &str) == CUDA_SUCCESS)
        return str;
      else
        return unrecognizedStr; // shouldn't happen
    }

    cudaError_t CUDARTAPI cudaGetDeviceCount(int* count)
    {
      CUresult result = cuDeviceGetCount(count);
      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaGetDeviceProperties(cudaDeviceProp* prop, int device)
    {
      CUdevice deviceHandle;
      CUresult result = cuDeviceGet(&deviceHandle, device);
      if (result != CUDA_SUCCESS)
        return Runtime::setError(result);

      cudaDeviceProp p;
      memset(&p, 0, sizeof(p));

      int sharedMemPerBlock, memPitch, totalConstMem,
          textureAlignment, texturePitchAlignment, surfaceAlignment,
          sharedMemPerMultiprocessor, sharedMemPerBlockOptin, reservedSharedMemPerBlock;

      const std::vector<std::pair<int*, CUdevice_attribute>> attribs =
      {
        {&sharedMemPerBlock,                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK},
        {&p.regsPerBlock,                   CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK},
        {&p.warpSize,                       CU_DEVICE_ATTRIBUTE_WARP_SIZE},
        {&memPitch,                         CU_DEVICE_ATTRIBUTE_MAX_PITCH},
        {&p.maxThreadsPerBlock,             CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK},
        {&p.maxThreadsDim[0],               CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X},
        {&p.maxThreadsDim[1],               CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y},
        {&p.maxThreadsDim[2],               CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z},
        {&p.maxGridSize[0],                 CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X},
        {&p.maxGridSize[1],                 CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y},
        {&p.maxGridSize[2],                 CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z},
      //{&p.clockRate,                      CU_DEVICE_ATTRIBUTE_CLOCK_RATE}, // deprecated
        {&totalConstMem,                    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY},
        {&p.major,                          CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR},
        {&p.minor,                          CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR},
        {&textureAlignment,                 CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT},
        {&texturePitchAlignment,            CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT},
      //{&p.deviceOverlap,                  CU_DEVICE_ATTRIBUTE_GPU_OVERLAP},
        {&p.multiProcessorCount,            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT},
      //{&p.kernelExecTimeoutEnabled,       CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT},
        {&p.integrated,                     CU_DEVICE_ATTRIBUTE_INTEGRATED},
        {&p.canMapHostMemory,               CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY},
      //{&p.computeMode,                    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE},
        {&p.maxTexture1D,                   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH},
        {&p.maxTexture1DMipmap,             CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH},
      //{&p.maxTexture1DLinear,             CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH},
        {&p.maxTexture2D[0],                CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH},
        {&p.maxTexture2D[1],                CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT},
        {&p.maxTexture2DMipmap[0],          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH},
        {&p.maxTexture2DMipmap[1],          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT},
        {&p.maxTexture2DLinear[0],          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH},
        {&p.maxTexture2DLinear[1],          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT},
        {&p.maxTexture2DLinear[2],          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH},
        {&p.maxTexture2DGather[0],          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH},
        {&p.maxTexture2DGather[1],          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT},
        {&p.maxTexture3D[0],                CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH},
        {&p.maxTexture3D[1],                CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT},
        {&p.maxTexture3D[2],                CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH},
        {&p.maxTexture3DAlt[0],             CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE},
        {&p.maxTexture3DAlt[1],             CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE},
        {&p.maxTexture3DAlt[2],             CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE},
        {&p.maxTextureCubemap,              CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH},
        {&p.maxTexture1DLayered[0],         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH},
        {&p.maxTexture1DLayered[1],         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS},
        {&p.maxTexture2DLayered[0],         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH},
        {&p.maxTexture2DLayered[1],         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT},
        {&p.maxTexture2DLayered[2],         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS},
        {&p.maxTextureCubemapLayered[0],    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH},
        {&p.maxTextureCubemapLayered[1],    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS},
        {&p.maxSurface1D,                   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH},
        {&p.maxSurface2D[0],                CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH},
        {&p.maxSurface2D[1],                CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT},
        {&p.maxSurface3D[0],                CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH},
        {&p.maxSurface3D[1],                CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT},
        {&p.maxSurface3D[2],                CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH},
        {&p.maxSurface1DLayered[0],         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH},
        {&p.maxSurface1DLayered[1],         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS},
        {&p.maxSurface2DLayered[0],         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH},
        {&p.maxSurface2DLayered[1],         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT},
        {&p.maxSurface2DLayered[2],         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS},
        {&p.maxSurfaceCubemap,              CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH},
        {&p.maxSurfaceCubemapLayered[0],    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH},
        {&p.maxSurfaceCubemapLayered[1],    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS},
        {&surfaceAlignment,                 CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT},
        {&p.concurrentKernels,              CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS},
        {&p.ECCEnabled,                     CU_DEVICE_ATTRIBUTE_ECC_ENABLED},
        {&p.pciBusID,                       CU_DEVICE_ATTRIBUTE_PCI_BUS_ID},
        {&p.pciDeviceID,                    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID},
        {&p.pciDomainID,                    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID},
        {&p.tccDriver,                      CU_DEVICE_ATTRIBUTE_TCC_DRIVER},
        {&p.asyncEngineCount,               CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT},
        {&p.unifiedAddressing,              CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING},
      //{&p.memoryClockRate,                CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE},
        {&p.memoryBusWidth,                 CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH},
        {&p.l2CacheSize,                    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE},
        {&p.persistingL2CacheMaxSize,       CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE},
        {&p.maxThreadsPerMultiProcessor,    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR},
        {&p.streamPrioritiesSupported,      CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED},
        {&p.globalL1CacheSupported,         CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED},
        {&p.localL1CacheSupported,          CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED},
        {&sharedMemPerMultiprocessor,       CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR},
        {&p.regsPerMultiprocessor,          CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR},
        {&p.managedMemory,                  CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY},
        {&p.isMultiGpuBoard,                CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD},
        {&p.multiGpuBoardGroupID,           CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID},
        {&p.hostNativeAtomicSupported,      CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED},
      //{&p.singleToDoublePrecisionPerfRatio, CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO},
        {&p.pageableMemoryAccess,           CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS},
        {&p.concurrentManagedAccess,        CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS},
        {&p.computePreemptionSupported,     CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED},
        {&p.canUseHostPointerForRegisteredMem, CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM},
        {&p.cooperativeLaunch,              CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH},
      //{&p.cooperativeMultiDeviceLaunch,   CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH},
        {&sharedMemPerBlockOptin,           CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN},
        {&p.pageableMemoryAccessUsesHostPageTables, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES},
        {&p.directManagedMemAccessFromHost, CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST},
        {&p.maxBlocksPerMultiProcessor,     CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR},
        {&p.accessPolicyMaxWindowSize,      CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE},
        {&reservedSharedMemPerBlock,        CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK},
      #if CUDA_VERSION >= 12000
        {&p.hostRegisterSupported,          CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED},
        {&p.sparseCudaArraySupported,       CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED},
        {&p.hostRegisterReadOnlySupported,  CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED},
        {&p.timelineSemaphoreInteropSupported, CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED},
        {&p.memoryPoolsSupported,           CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED},
        {&p.gpuDirectRDMASupported,         CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED},
        {(int*)&p.gpuDirectRDMAFlushWritesOptions, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS},
        {&p.gpuDirectRDMAWritesOrdering,    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING},
        {(int*)&p.memoryPoolSupportedHandleTypes, CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES},
        {&p.deferredMappingCudaArraySupported, CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED},
        {&p.ipcEventSupported,              CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED},
        {&p.clusterLaunch,                  CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH},
        {&p.unifiedFunctionPointers,        CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS},
      #endif
      };

      for (const auto& attrib : attribs)
      {
        result = cuDeviceGetAttribute(attrib.first, attrib.second, deviceHandle);
        if (result != CUDA_SUCCESS)
          return Runtime::setError(result);
      }

      p.sharedMemPerBlock          = sharedMemPerBlock;
      p.memPitch                   = memPitch;
      p.totalConstMem              = totalConstMem;
      p.textureAlignment           = textureAlignment;
      p.texturePitchAlignment      = texturePitchAlignment;
      p.surfaceAlignment           = surfaceAlignment;
      p.sharedMemPerMultiprocessor = sharedMemPerMultiprocessor;
      p.sharedMemPerBlockOptin     = sharedMemPerBlockOptin;
      p.reservedSharedMemPerBlock  = reservedSharedMemPerBlock;

      result = cuDeviceGetName(p.name, sizeof(p.name), deviceHandle);
      if (result != CUDA_SUCCESS)
        return Runtime::setError(result);

    #if CUDA_VERSION >= 12000
      result = cuDeviceGetUuid_v2(&p.uuid, deviceHandle);
    #else
      result = cuDeviceGetUuid(&p.uuid, deviceHandle);
    #endif
      if (result != CUDA_SUCCESS)
        return Runtime::setError(result);

    #if defined(_WIN32)
      result = cuDeviceGetLuid(p.luid, &p.luidDeviceNodeMask, deviceHandle);
      if (result != CUDA_SUCCESS && result != CUDA_ERROR_NOT_SUPPORTED)
        return Runtime::setError(result);
    #endif

      result = cuDeviceTotalMem(&p.totalGlobalMem, deviceHandle);
      if (result != CUDA_SUCCESS)
        return Runtime::setError(result);

      *prop = p;
      return cudaSuccess;
    }

    cudaError_t CUDARTAPI cudaStreamAddCallback(cudaStream_t stream,
                                                cudaStreamCallback_t callback,
                                                void* userData,
                                                unsigned int flags)
    {
      StreamCallbackDesc* desc = new StreamCallbackDesc{callback, userData};
      CUresult result = cuStreamAddCallback(stream, Runtime::streamCallback, desc, 0);
      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream)
    {
      CUresult result = cuStreamSynchronize(stream);
      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value)
    {
      CUfunction funcHandle;
      CUresult result = Runtime::get().getFunction(func, funcHandle);
      if (result != CUDA_SUCCESS)
        return Runtime::setError(result);

      CUfunction_attribute cuAttr;
      switch (attr)
      {
      case cudaFuncAttributeMaxDynamicSharedMemorySize:
        cuAttr = CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        break;
      case cudaFuncAttributePreferredSharedMemoryCarveout:
        cuAttr = CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
        break;
      case cudaFuncAttributeRequiredClusterWidth:
        cuAttr = CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH;
        break;
      case cudaFuncAttributeRequiredClusterHeight:
        cuAttr = CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT;
        break;
      case cudaFuncAttributeRequiredClusterDepth:
        cuAttr = CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH;
        break;
      case cudaFuncAttributeClusterSchedulingPolicyPreference:
        cuAttr = CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
        break;
      default:
        return Runtime::setError(cudaErrorInvalidValue);
      }

      result = cuFuncSetAttribute(funcHandle, cuAttr, value);
      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaLaunchKernel(const void* func,
                                          dim3 gridDim,
                                          dim3 blockDim,
                                          void** args,
                                          size_t sharedMem,
                                          cudaStream_t stream)
    {
      CUfunction funcHandle;
      CUresult result = Runtime::get().getFunction(func, funcHandle);
      if (result != CUDA_SUCCESS)
        return Runtime::setError(result);

      result = cuLaunchKernel(funcHandle,
                              gridDim.x, gridDim.y, gridDim.z,
                              blockDim.x, blockDim.y, blockDim.z,
                              sharedMem,
                              stream,
                              args,
                              nullptr);

      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaMallocManaged(void** devPtr, size_t size, unsigned int flags)
    {
      unsigned int cuFlags = 0;
      if (flags & cudaMemAttachGlobal)
        cuFlags |= CU_MEM_ATTACH_GLOBAL;
      if (flags & cudaMemAttachHost)
        cuFlags |= CU_MEM_ATTACH_HOST;
      if (flags & cudaMemAttachSingle)
        cuFlags |= CU_MEM_ATTACH_SINGLE;

      CUresult result = cuMemAllocManaged((CUdeviceptr*)devPtr, size, cuFlags);
      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaMalloc(void** devPtr, size_t size)
    {
      CUresult result = cuMemAlloc((CUdeviceptr*)devPtr, size);
      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaMallocHost(void** ptr, size_t size)
    {
      CUresult result = cuMemAllocHost(ptr, size);
      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaFree(void* devPtr)
    {
      CUresult result = cuMemFree((CUdeviceptr)devPtr);
      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaFreeHost(void* ptr)
    {
      CUresult result = cuMemFreeHost(ptr);
      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
    {
      CUresult result;

      switch (kind)
      {
      case cudaMemcpyDefault:
        result = cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, count);
        break;
      case cudaMemcpyHostToHost:
        memcpy(dst, src, count);
        result = CUDA_SUCCESS;
        break;
      case cudaMemcpyHostToDevice:
        result = cuMemcpyHtoD((CUdeviceptr)dst, src, count);
        break;
      case cudaMemcpyDeviceToHost:
        result = cuMemcpyDtoH(dst, (CUdeviceptr)src, count);
        break;
      case cudaMemcpyDeviceToDevice:
        result = cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, count);
        break;
      default:
        return Runtime::setError(cudaErrorInvalidMemcpyDirection);
      }

      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind,
                                          cudaStream_t stream)
    {
      CUresult result;

      switch (kind)
      {
      case cudaMemcpyDefault:
        result = cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, stream);
        break;
      case cudaMemcpyHostToHost:
        memcpy(dst, src, count); // FIXME: not async
        result = CUDA_SUCCESS;
        break;
      case cudaMemcpyHostToDevice:
        result = cuMemcpyHtoDAsync((CUdeviceptr)dst, src, count, stream);
        break;
      case cudaMemcpyDeviceToHost:
        result = cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, count, stream);
        break;
      case cudaMemcpyDeviceToDevice:
        result = cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, stream);
        break;
      default:
        return Runtime::setError(cudaErrorInvalidMemcpyDirection);
      }

      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream)
    {
      CUresult result = cuMemsetD8Async((CUdeviceptr)devPtr, (unsigned char)value, count, stream);
      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr)
    {
      CUmemorytype type;
      unsigned int isManaged;

      CUpointer_attribute cuAttributes[] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                            CU_POINTER_ATTRIBUTE_IS_MANAGED,
                                            CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                            CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                            CU_POINTER_ATTRIBUTE_HOST_POINTER};

      void* cuAttributeValues[] = {&type,
                                  &isManaged,
                                  &attributes->device,
                                  &attributes->devicePointer,
                                  &attributes->hostPointer};

      CUresult result = cuPointerGetAttributes(5, cuAttributes, cuAttributeValues, (CUdeviceptr)ptr);
      if (result != CUDA_SUCCESS)
        return Runtime::setError(result);

      if (isManaged)
      {
        attributes->type = cudaMemoryTypeManaged;
      }
      else
      {
        switch (type)
        {
        case CU_MEMORYTYPE_HOST:
        case CU_MEMORYTYPE_UNIFIED: // shouldn't happen, but just in case
          attributes->type = cudaMemoryTypeHost;
          break;
        case CU_MEMORYTYPE_DEVICE:
          attributes->type = cudaMemoryTypeDevice;
          break;
        default: // should be 0
          attributes->type = cudaMemoryTypeUnregistered;
        }
      }

      return cudaSuccess;
    }

    cudaError_t CUDARTAPI cudaImportExternalMemory(cudaExternalMemory_t* extMem_out,
                                                  const cudaExternalMemoryHandleDesc* memHandleDesc)
    {
      CUDA_EXTERNAL_MEMORY_HANDLE_DESC cuMemHandleDesc{};

      switch (memHandleDesc->type)
      {
      case cudaExternalMemoryHandleTypeOpaqueFd:
        cuMemHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        break;
      case cudaExternalMemoryHandleTypeOpaqueWin32:
        cuMemHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
        break;
      case cudaExternalMemoryHandleTypeOpaqueWin32Kmt:
        cuMemHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT;
        break;
      case cudaExternalMemoryHandleTypeD3D12Heap:
        cuMemHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP;
        break;
      case cudaExternalMemoryHandleTypeD3D12Resource:
        cuMemHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
        break;
      case cudaExternalMemoryHandleTypeD3D11Resource:
        cuMemHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE;
        break;
      case cudaExternalMemoryHandleTypeD3D11ResourceKmt:
        cuMemHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT;
        break;
      case cudaExternalMemoryHandleTypeNvSciBuf:
        cuMemHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF;
        break;
      default:
        return Runtime::setError(cudaErrorInvalidValue);
      }

      switch (memHandleDesc->type)
      {
      case cudaExternalMemoryHandleTypeOpaqueFd:
        cuMemHandleDesc.handle.fd = memHandleDesc->handle.fd;
        break;
      case cudaExternalMemoryHandleTypeNvSciBuf:
        cuMemHandleDesc.handle.nvSciBufObject = memHandleDesc->handle.nvSciBufObject;
        break;
      default:
        cuMemHandleDesc.handle.win32.handle = memHandleDesc->handle.win32.handle;
        cuMemHandleDesc.handle.win32.name   = memHandleDesc->handle.win32.name;
        break;
      }

      cuMemHandleDesc.size = memHandleDesc->size;

      cuMemHandleDesc.flags = 0;
      if (memHandleDesc->flags & cudaExternalMemoryDedicated)
        cuMemHandleDesc.flags |= CUDA_EXTERNAL_MEMORY_DEDICATED;

      CUresult result = cuImportExternalMemory((CUexternalMemory*)extMem_out, &cuMemHandleDesc);
      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaExternalMemoryGetMappedBuffer(
                            void** devPtr,
                            cudaExternalMemory_t extMem,
                            const cudaExternalMemoryBufferDesc* bufferDesc)
    {
      CUDA_EXTERNAL_MEMORY_BUFFER_DESC cuBufferDesc{};
      cuBufferDesc.offset = bufferDesc->offset;
      cuBufferDesc.size   = bufferDesc->size;
      cuBufferDesc.flags  = 0;

      CUresult result = cuExternalMemoryGetMappedBuffer((CUdeviceptr*)devPtr,
                                                        (CUexternalMemory)extMem,
                                                        &cuBufferDesc);
      return Runtime::setError(result);
    }

    cudaError_t CUDARTAPI cudaDestroyExternalMemory(cudaExternalMemory_t extMem)
    {
      CUresult result = cuDestroyExternalMemory((CUexternalMemory)extMem);
      return Runtime::setError(result);
    }
  } // extern "C"
} // namespace curtn