#include "vulkan/vulkan.h"

#include <vector>
#include <optional>
#include <set>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>

#include "OpenImageIO/imageio.h"

#include "OpenImageDenoise/oidn.hpp"

const char* vkResultString(VkResult r) {
    switch(r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
        case VK_PIPELINE_COMPILE_REQUIRED: return "VK_PIPELINE_COMPILE_REQUIRED";
        case VK_ERROR_NOT_PERMITTED: return "VK_ERROR_NOT_PERMITTED";
        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
        case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";
        case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
        case VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR: return "VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR";
        case VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR: return "VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR";
        case VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR: return "VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR";
        case VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR: return "VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR";
        case VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR: return "VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR";
        case VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR: return "VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR";
        case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT: return "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
        case VK_THREAD_IDLE_KHR: return "VK_THREAD_IDLE_KHR";
        case VK_THREAD_DONE_KHR: return "VK_THREAD_DONE_KHR";
        case VK_OPERATION_DEFERRED_KHR: return "VK_OPERATION_DEFERRED_KHR";
        case VK_OPERATION_NOT_DEFERRED_KHR: return "VK_OPERATION_NOT_DEFERRED_KHR";
        case VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR: return "VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR";
        case VK_ERROR_COMPRESSION_EXHAUSTED_EXT: return "VK_ERROR_COMPRESSION_EXHAUSTED_EXT";
        case VK_INCOMPATIBLE_SHADER_BINARY_EXT: return "VK_INCOMPATIBLE_SHADER_BINARY_EXT";
        case VK_PIPELINE_BINARY_MISSING_KHR: return "VK_PIPELINE_BINARY_MISSING_KHR";
        case VK_ERROR_NOT_ENOUGH_SPACE_KHR: return "VK_ERROR_NOT_ENOUGH_SPACE_KHR";
        default: 
            return "Unknown Vulkan result";
    }
}

#define CHECK_VULKAN(FN)                                   \
    {                                                      \
        VkResult r = FN;                                   \
        if (r != VK_SUCCESS) {                             \
            std::cout << #FN << " failed with error " << vkResultString(r) << "\n" << std::flush; \
            throw std::runtime_error(#FN " failed!");      \
        }                                                  \
    }


bool VERBOSE = false;

const std::vector<const char*> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"
};

VKAPI_ATTR VkBool32 VKAPI_CALL DEBUG_CALLBACK(  VkDebugUtilsMessageSeverityFlagBitsEXT message_severity, 
                                                VkDebugUtilsMessageTypeFlagsEXT message_type, 
                                                const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data, 
                                                void* p_user_data) 
{
    if (VERBOSE) std::cerr << "validation layer: " << p_callback_data->pMessage << std::endl;
    return VK_FALSE;
}

#ifdef _WIN32
PFN_vkGetMemoryWin32HandleKHR GetMemoryWin32HandleKHR = nullptr;
#else
PFN_vkGetMemoryFdKHR GetMemoryFdKHR = nullptr;
#endif

class Instance {
public:
    Instance() {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "vkdlss";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "No Engine";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app_info;

        std::vector<const char*> extensions {
            "VK_KHR_device_group_creation",
            "VK_KHR_get_physical_device_properties2", 
        };

#if NDEBUG
        if (std::find_if(extensions.begin(), extensions.end(), [](const char* e) { return strcmp(e, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0; }) == extensions.end()) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
#endif

        instance_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        instance_info.ppEnabledExtensionNames = extensions.data();

#ifndef NDEBUG 
        VkDebugUtilsMessengerCreateInfoEXT dbgmsg_info{};
        dbgmsg_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbgmsg_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dbgmsg_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dbgmsg_info.pfnUserCallback = DEBUG_CALLBACK;

        instance_info.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
        instance_info.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        instance_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &dbgmsg_info;
#else
        instance_info.enabledLayerCount = 0;
        instance_info.pNext = nullptr;
#endif

        CHECK_VULKAN(vkCreateInstance(&instance_info, nullptr, &instance));
    }
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = delete;
    Instance& operator=(Instance&&) = delete;
    ~Instance() {
        vkDestroyInstance(instance, nullptr);
    }

public:
    operator VkInstance() const { return get(); }
    VkInstance get() const { return instance; }

private:
    VkInstance instance = nullptr;
};

const std::vector<const char*> DEVICE_EXTENSIONS = {
#ifdef _WIN32
    VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
#else
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
#endif
};


class Device {
public:
    Device(Instance& instance) {

        uint32_t device_count = 0;
        CHECK_VULKAN(vkEnumeratePhysicalDevices(instance, &device_count, nullptr));

        if (device_count == 0) {
            throw std::runtime_error("could not find GPU, that supports vulkan");
        }

        std::vector<VkPhysicalDevice> devices(device_count);
        CHECK_VULKAN(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()));

        for (const auto& d : devices) {

            // device vendor has do be NVidia
            VkPhysicalDeviceProperties properties;
            vkGetPhysicalDeviceProperties(d, &properties);

            // check extension support
            uint32_t count;
            CHECK_VULKAN(vkEnumerateDeviceExtensionProperties(d, nullptr, &count, nullptr));

            std::vector<VkExtensionProperties> available_extensions(count);
            CHECK_VULKAN(vkEnumerateDeviceExtensionProperties(d, nullptr, &count, available_extensions.data()));


            std::set<std::string> required_extensions(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());

            for (const auto& extension : available_extensions) {
                required_extensions.erase(extension.extensionName);
            }

            if (!required_extensions.empty()) {
                continue;
            }

            // device OK
            physical_device = d;
            break; 
        }

        if (!physical_device) {
            throw std::runtime_error("could not find GPU, that supports vulkan");
        }



        uint32_t index = get_graphics_and_compute_queue_index();

        std::vector<VkDeviceQueueCreateInfo> queue_create_infos;

        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_create_info{};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = get_graphics_and_compute_queue_index();
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;
        queue_create_infos.push_back(queue_create_info);


        VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features;
        buffer_device_address_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        buffer_device_address_features.bufferDeviceAddress = VK_TRUE;
        buffer_device_address_features.pNext = nullptr;

        VkPhysicalDeviceFeatures2 device_features2{};
        device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        device_features2.pNext = &buffer_device_address_features;

        VkDeviceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
        create_info.pQueueCreateInfos = queue_create_infos.data();

        create_info.pEnabledFeatures = nullptr;
        create_info.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
        create_info.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
        create_info.pNext = &device_features2;

#ifndef NDEBUG
        create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
        create_info.ppEnabledLayerNames = validation_layers.data();
#else
        create_info.enabledLayerCount = 0;
#endif

        CHECK_VULKAN(vkCreateDevice(physical_device, &create_info, nullptr, &device));

        vkGetDeviceQueue(device, index, 0, &queue);

#ifdef _WIN32
        GetMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
        if (!GetMemoryWin32HandleKHR) {
            throw std::runtime_error("could not load GetMemoryWin32HandleKHR");
        }
#else
        GetMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
        if (!GetMemoryFdKHR) {
            throw std::runtime_error("could not load GetMemoryFdKHR");
        }
#endif

        uint32_t instance_version;
        CHECK_VULKAN(vkEnumerateInstanceVersion(&instance_version));
        VkPhysicalDeviceProperties device_properties;
        vkGetPhysicalDeviceProperties(physical_device, &device_properties);
        std::cout << std::setw(15) << std::left << "Vulkan Device: " << std::setw(30) << std::left << device_properties.deviceName 
            << "Version: " << VK_VERSION_MAJOR(instance_version) << "."
                            << VK_VERSION_MINOR(instance_version) << "."
                            << VK_VERSION_PATCH(instance_version) << std::endl;
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;
    ~Device() {
        vkDestroyDevice(device, nullptr);
    }

public:
    operator VkDevice() const { return get(); }
    VkDevice get() const { return device; }

    VkPhysicalDevice get_pdevice() { return physical_device; }

    VkQueue get_queue() const { return queue; }


    uint32_t get_graphics_and_compute_queue_index() const {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);

        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());

        int i = 0;
        for (const auto& f : families) {
            if ((f.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (f.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                return i;
            }
        }

        throw std::runtime_error("Did not find a queue family index that has VK_QUEUE_GRAPHICS_BIT and VK_QUEUE_COMPUTE_BIT");
    }

private:
    VkPhysicalDevice physical_device = nullptr;
    VkDevice device = nullptr;
    VkQueue queue = nullptr;
};



class CommandBuffer {
public:
    CommandBuffer(VkDevice device, VkCommandPool pool): device(device), pool(pool) {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        CHECK_VULKAN(vkAllocateCommandBuffers(device, &alloc_info, &buffer));

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        CHECK_VULKAN(vkBeginCommandBuffer(buffer, &begin_info));
    }
    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;
    CommandBuffer(CommandBuffer&& other): buffer(other.buffer) { 
        other.buffer = nullptr;
    }
    CommandBuffer& operator=(CommandBuffer&& other) {
        buffer = other.buffer;
        other.buffer = nullptr;
        return *this;
    }
    ~CommandBuffer() {
        if (buffer) vkFreeCommandBuffers(device, pool, 1, &buffer);
    }

public:
    operator VkCommandBuffer() const { return get(); }
    VkCommandBuffer get() const { return buffer; }

    void end_submit(VkQueue queue = nullptr, bool wait = true) {
        CHECK_VULKAN(vkEndCommandBuffer(buffer));

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &buffer;

        CHECK_VULKAN(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE));
        CHECK_VULKAN(vkQueueWaitIdle(queue));
    }


private:
    VkDevice device;
    VkCommandBuffer buffer = nullptr;
    VkCommandPool pool = nullptr;
};


class CommandPool {
public:
    CommandPool(Device &device): device(device) {
        uint32_t index = device.get_graphics_and_compute_queue_index();

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = index;

        CHECK_VULKAN(vkCreateCommandPool(device, &pool_info, nullptr, &pool));
    }
    CommandPool(const CommandPool&) = delete;
    CommandPool& operator=(const CommandPool&) = delete;
    CommandPool(CommandPool&&) = delete;
    CommandPool& operator=(CommandPool&&) = delete;
    ~CommandPool() {
        vkDestroyCommandPool(device, pool, nullptr);
    }

public:
    operator VkCommandPool() const { return get(); }
    VkCommandPool get() const { return pool; }

    CommandBuffer begin() {
        return CommandBuffer(device, pool);
    }

private:
    VkDevice device;
    VkCommandPool pool = nullptr;
};



struct ExtMemType {
    oidn::ExternalMemoryTypeFlag oidn;
    VkExternalMemoryHandleTypeFlagBits vk;

    ExtMemType(oidn::DeviceRef& oidn_device) {
        // Find a compatible external memory handle type
        const auto oidn_external_mem_types = oidn_device.get<oidn::ExternalMemoryTypeFlags>("externalMemoryTypes");

        if (oidn_external_mem_types & oidn::ExternalMemoryTypeFlag::OpaqueFD) {
            oidn = oidn::ExternalMemoryTypeFlag::OpaqueFD;
            vk = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        } else if (oidn_external_mem_types & oidn::ExternalMemoryTypeFlag::DMABuf) {
            oidn = oidn::ExternalMemoryTypeFlag::DMABuf;
            vk = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        } else if (oidn_external_mem_types & oidn::ExternalMemoryTypeFlag::OpaqueWin32) {
            oidn = oidn::ExternalMemoryTypeFlag::OpaqueWin32;
            vk = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        } else {
            throw std::runtime_error("failed to find compatible external memory type");
        }
    }
};

std::ostream& operator <<(std::ostream& os, ExtMemType ext_mem_type)
{

    os << std::setw(15) << std::left << "Ext Mem Type (oidn | vk): ";
    switch (ext_mem_type.oidn) {
        case oidn::ExternalMemoryTypeFlag::OpaqueFD:    os << "OpaqueFD"; break;
        case oidn::ExternalMemoryTypeFlag::DMABuf:      os << "DMABuf"; break;
        case oidn::ExternalMemoryTypeFlag::OpaqueWin32: os << "OpaqueWin32"; break;
        default: os << "Unknown"; break;
    }
    switch (ext_mem_type.vk) {
        case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT:      os << " | OpaqueFD"; break;
        case VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT:    os << " | DMABuf"; break;
        case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT:   os << " | OpaqueWin32"; break;
        default: os << "Unknown"; break;
    }

    return os;
  }


class Buffer {
public:
    Buffer(Device& device, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props, std::optional<ExtMemType> ext_mem_type = {}): physical_device(device.get_pdevice()), device(device), size(size), ext_mem_type(ext_mem_type) {

        // create buffer
        {
            VkBufferCreateInfo create_info = {};
            create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            create_info.size = size;
            create_info.usage = usage;
            create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VkExternalMemoryBufferCreateInfo external_mem_create_info{};
            if (ext_mem_type) {
                external_mem_create_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
                external_mem_create_info.handleTypes = ext_mem_type->vk;

                create_info.pNext = &external_mem_create_info;
            }

            CHECK_VULKAN(vkCreateBuffer(device, &create_info, nullptr, &buf));
        }

        // alloc mem 
        {
            VkMemoryRequirements mem_reqs = {};
            vkGetBufferMemoryRequirements(device, buf, &mem_reqs);

            VkMemoryAllocateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            info.allocationSize = size;
            info.memoryTypeIndex = find_memory_type_index(mem_reqs.memoryTypeBits, mem_props);

            VkMemoryAllocateFlagsInfo flags = {};
            if (mem_props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
                info.pNext = &flags;
            }

            VkExportMemoryAllocateInfo export_info{};
            if (ext_mem_type) {
                export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
                export_info.handleTypes = ext_mem_type->vk;
                export_info.pNext = info.pNext;

                info.pNext = &export_info;
            }

            CHECK_VULKAN(vkAllocateMemory(device, &info, nullptr, &mem));
        }

        CHECK_VULKAN(vkBindBufferMemory(device, buf, mem, 0));
    }

    static Buffer make_image_buffer(Device& device, oidn::DeviceRef& oidn_device, uint32_t width, uint32_t height, oidn::Format format) {
        ExtMemType ext_mem_type(oidn_device);

        size_t size_bytes = 0;
        switch (format) {
            case oidn::Format::Undefined: size_bytes = 0; break;
            case oidn::Format::Float:     size_bytes = sizeof(float); break;
            case oidn::Format::Float2:    size_bytes = sizeof(float)*2; break;
            case oidn::Format::Float3:    size_bytes = sizeof(float)*3; break; 
            case oidn::Format::Float4:    size_bytes = sizeof(float)*4; break;
            case oidn::Format::Half:      size_bytes = sizeof(int16_t); break;
            case oidn::Format::Half2:     size_bytes = sizeof(int16_t)*2; break;
            case oidn::Format::Half3:     size_bytes = sizeof(int16_t)*3; break;
            case oidn::Format::Half4:     size_bytes = sizeof(int16_t)*4; break;
            default:
                throw std::invalid_argument("invalid format");
        }

        size_bytes = size_bytes * width * height;

        auto buffer = Buffer(device, size_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ext_mem_type);

        return buffer;
    }

    static Buffer make_staging_buffer(Device& device, size_t size) {
        return Buffer(device, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    Buffer(Buffer&) = delete;
    Buffer operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept 
        : physical_device(other.physical_device), 
          device(other.device), 
          size(other.size), 
          buf(other.buf), 
          mem(other.mem) 
    {
        other.buf = VK_NULL_HANDLE;
        other.mem = VK_NULL_HANDLE;
    }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            if (buf) vkDestroyBuffer(device, buf, nullptr);
            if (mem) vkFreeMemory(device, mem, nullptr);

            physical_device = other.physical_device;
            device = other.device;
            size = other.size;
            buf = other.buf;
            mem = other.mem;

            other.buf = VK_NULL_HANDLE;
            other.mem = VK_NULL_HANDLE;
        }
        return *this;
    }


    ~Buffer() {
        std::cout << "delete buffer"
                    << "    buf   " << buf   
                    << "    mem   " << mem << std::endl;
        if (buf) vkDestroyBuffer(device, buf, nullptr);
        if (mem) vkFreeMemory(device, mem, nullptr);
    }

private:
    uint32_t find_memory_type_index(uint32_t type_filter, VkMemoryPropertyFlags props) const
    {
        VkPhysicalDeviceMemoryProperties device_mem_props = {};
        vkGetPhysicalDeviceMemoryProperties(physical_device, &device_mem_props);
        for (uint32_t i = 0; i < device_mem_props.memoryTypeCount; ++i) {
            if (type_filter & (1 << i) &&
                (device_mem_props.memoryTypes[i].propertyFlags & props) == props) {
                return i;
            }
        }
        throw std::runtime_error("failed to find memory type index");
    }


public:
    size_t get_size() {
        return size;
    }
    VkBuffer get_buf() {
        return buf;
    }
    VkDeviceMemory get_mem() {
        return mem;
    }
    std::optional<ExtMemType> get_ext_mem_type() {
        return ext_mem_type;
    }

    #ifdef _WIN32
    HANDLE get_ext_mem_handle()
    {
        HANDLE handle = NULL;
        VkMemoryGetWin32HandleInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        info.memory = mem;
        info.handleType = ext_mem_type->vk;
        CHECK_VULKAN(GetMemoryWin32HandleKHR(device, &info, &handle));
        return handle;
    }
    #else
    int get_ext_mem_handle()
    {
        int fd = 0;
        VkMemoryGetFdInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        info.memory = mem;
        info.handleType = ext_mem_type->vk;
        CHECK_VULKAN(GetMemoryFdKHR(device, &info, &fd));
        return fd;
    }
    #endif

private:
    friend class ImageBuffer;

    VkPhysicalDevice physical_device;
    VkDevice device;

    size_t size;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;

    std::optional<ExtMemType> ext_mem_type;
};



class ImageBuffer {
public:
    ImageBuffer(Device& device, oidn::DeviceRef& oidn_device, uint32_t width, uint32_t height, oidn::Format format)
        : buffer(Buffer::make_image_buffer(device, oidn_device, width, height, format)),
          width(width), height(height), format(format) {}

    ImageBuffer(const ImageBuffer&) = delete;
    ImageBuffer& operator=(const ImageBuffer&) = delete;
    ImageBuffer(ImageBuffer&&) = default;
    ImageBuffer& operator=(ImageBuffer&&) = default;

    ~ImageBuffer() = default;


public:
    static ImageBuffer load_exr(std::string filename, Device& device, oidn::DeviceRef& oidn_device, CommandPool& cp) {
        auto in = OIIO::ImageInput::open(filename);
        if (!in) {
            throw std::runtime_error("failed to open image file");
        }

        OIIO::ImageSpec spec = in->spec();
        if ((spec.channelformat(0) == OIIO::TypeDesc::FLOAT || spec.channelformat(0) == OIIO::TypeDesc::HALF) && (spec.nchannels < 1 || spec.nchannels > 4)) {
            throw std::invalid_argument("unsupported format");
        }

        int width = spec.width;
        int height = spec.height;
        int nchannels = spec.nchannels;
        std::vector<oidn::Format> formats = {oidn::Format::Float, oidn::Format::Float2, oidn::Format::Float3, oidn::Format::Float4,
                                                oidn::Format::Half, oidn::Format::Half2, oidn::Format::Half3, oidn::Format::Half4};
        oidn::Format format = formats[spec.channelformat(0) == OIIO::TypeDesc::FLOAT ? nchannels - 1 : nchannels + 3];


        ImageBuffer buffer(device, oidn_device, width, height, format);
        Buffer staging = Buffer::make_staging_buffer(device, buffer.buffer.size);


        void* mapped;
        CHECK_VULKAN(vkMapMemory(device, staging.mem, 0, staging.size, 0, &mapped));
        in->read_image(0, 0, 0, nchannels, spec.channelformat(0), mapped);
        vkUnmapMemory(device, staging.mem);
        in->close();

        auto cb = cp.begin();

        VkBufferCopy copyRegion = {};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = staging.size;
        vkCmdCopyBuffer(cb, staging.buf, buffer.buffer.buf, 1, &copyRegion);

        cb.end_submit(device.get_queue(), true);

        return buffer;
    }

    void save_exr(std::string filename, Device& device, CommandPool& cp) {
        // Create a staging buffer to copy the data to
        Buffer staging = Buffer::make_staging_buffer(device, buffer.size);

        auto cb = cp.begin();

        VkBufferCopy copyRegion = {};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = staging.size;
        vkCmdCopyBuffer(cb, buffer.buf, staging.buf, 1, &copyRegion);

        cb.end_submit(device.get_queue(), true);


        std::unique_ptr<OIIO::ImageOutput> out = OIIO::ImageOutput::create(filename);
        if (!out) {
            throw std::runtime_error("Could not create file: " + filename);
        }

        
        std::vector<oidn::Format> formats = {oidn::Format::Float, oidn::Format::Float2, oidn::Format::Float3, oidn::Format::Float4,
                                                oidn::Format::Half, oidn::Format::Half2, oidn::Format::Half3, oidn::Format::Half4};


        auto it = std::find(formats.begin(), formats.end(), format);
        if (it == formats.end()) {
            throw std::invalid_argument("unsupported format");
        }

        size_t nchannels = std::distance(formats.begin(), it) + 1;
        OIIO::TypeDesc type = OIIO::TypeDesc::FLOAT;

        if (nchannels > 4){
            nchannels = nchannels - 4;
            type = OIIO::TypeDesc::HALF;
        }

        OIIO::ImageSpec spec(width, height, nchannels, type);
        out->open(filename, spec);
        void* mapped;
        CHECK_VULKAN(vkMapMemory(device, staging.mem, 0, staging.size, 0, &mapped));
        out->write_image(type, mapped);
        vkUnmapMemory(device, staging.mem);
        out->close();
    }

    Buffer& get_buffer() {
        return buffer;
    }
    uint32_t get_width() {
        return width;
    }
    uint32_t get_height() {
        return height;
    }
    oidn::Format get_format() {
        return format;
    }

private:
    Buffer buffer;
    uint32_t width;
    uint32_t height;
    oidn::Format format;
};



void oidnErrorCallback(void* userPtr, oidn::Error error, const char* message)
{
  throw std::runtime_error(message);
}

std::string usage() {
    return
        "Usage:\n"
        "   vkdlss - run nvidia dlss on exr inputs\n"
        "\n"
        "SYNOPSIS\n"
        "   vkdlss --json <path> [OPTION]\n"
        "\n"
        "OPTIONS\n"
        "   --preset <name>  - preset name, accepts values in [Default, A - O], defalut = Default, refer to DLSS Programming Guide section 3.12 for description and dependencies\n"
        "   --quality <name> - performance quality level, accepts values in [perf, balanced, quality, ultra], defalut = ultra\n"
        "   --scale <float>  - scaling factor, accepts values in [1.0, 2.0], for quality ultra accepts value in [1.0, 3.0]\n"
        "   --out <ext>      - extension of the out files, i.e. <path>.out.<n>.exr, default = out\n"
        "   --help           - print this help\n"
        "   --verbose        - vebose console output\n"
        ;
}

bool has_param(const std::string& pname, std::vector<std::string>& args) {
    auto p = std::find(args.begin(), args.end(), pname);
    return p != args.end() && (p+1) != args.end();
}

std::string parse(const std::string& pname, std::vector<std::string>& args) {
    auto p = std::find(args.begin(), args.end(), pname);
    if (p == args.end()) {
        throw std::runtime_error("could not parse parameter: " + pname + "\n" + usage());
    }

    ++p;
    if (p == args.end()) {
        throw std::runtime_error("could not parse parameter: " + pname + "\n" + usage());
    }

    return *p;
}

std::string parse_opt(const std::string& pname, std::vector<std::string>& args, std::string def = "") {
    return has_param(pname, args)? parse(pname, args) : def;
}


int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv + 1, argv + argc);

    auto path_json = parse("--json", args);
    auto no_extension = path_json.substr(0, path_json.find_last_of("."));
    auto no_seqnum = no_extension.substr(0, no_extension.find_last_of("."));
    auto seqnum = no_extension.substr(no_extension.find_last_of(".") + 1);

    auto path_color = no_seqnum + ".hdr." + seqnum + ".exr";
    auto path_albedo = no_seqnum + ".alb1." + seqnum + ".exr";
    auto path_normal = no_seqnum + ".nrm1." + seqnum + ".exr";

    auto path_out = parse_opt("--out", args, "./out.exr");
    auto scale = std::stof(parse_opt("--scale", args, "2.0f"));


    std::cout << "oidn interop test" << std::endl;
    std::cout << "    color:  " << path_color << std::endl;
    std::cout << "    albedo: " << path_albedo << std::endl;
    std::cout << "    normal: " << path_normal << std::endl;
    std::cout << "    output: " << path_out << std::endl;
    std::cout << "    scale:  " << scale << std::endl;

    Instance instance;
    Device device(instance);
    CommandPool command_pool(device);


    // Query the UUID of the Vulkan physical device
    VkPhysicalDeviceIDProperties id_properties{};
    id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &id_properties;
    vkGetPhysicalDeviceProperties2(device.get_pdevice(), &properties);

    oidn::UUID uuid;
    std::memcpy(uuid.bytes, id_properties.deviceUUID, sizeof(uuid.bytes));

    // Initialize the denoiser device
    oidn::DeviceRef oidn_device = oidn::newDevice(uuid);
    if (oidn_device.getError() != oidn::Error::None) {
        throw std::runtime_error("Failed to create OIDN device.");
        exit(1);
    }
    oidn_device.commit();
    if (oidn_device.getError() != oidn::Error::None)
        throw std::runtime_error("Failed to commit OIDN device.");

    oidn::DeviceType deviceType = oidn_device.get<oidn::DeviceType>("type");
    const int versionMajor = oidn_device.get<int>("versionMajor");
    const int versionMinor = oidn_device.get<int>("versionMinor");
    const int versionPatch = oidn_device.get<int>("versionPatch");

    std::cout << std::setw(15) << std::left << "OIDN Device: ";
    switch (deviceType)
    {
    case oidn::DeviceType::Default: std::cout << std::setw(30) << std::left << "default"; break;
    case oidn::DeviceType::CPU:     std::cout << std::setw(30) << std::left << "CPU";     break;
    case oidn::DeviceType::SYCL:    std::cout << std::setw(30) << std::left << "SYCL";    break;
    case oidn::DeviceType::CUDA:    std::cout << std::setw(30) << std::left << "CUDA";    break;
    case oidn::DeviceType::HIP:     std::cout << std::setw(30) << std::left << "HIP";     break;
    case oidn::DeviceType::Metal:   std::cout << std::setw(30) << std::left << "Metal";   break;
    default:
      throw std::invalid_argument("invalid device type");
    }

    std::cout << "Version: " << versionMajor << "." << versionMinor << "." << versionPatch << std::endl;

    ImageBuffer color = ImageBuffer::load_exr(path_color, device, oidn_device, command_pool);
    ImageBuffer albedo = ImageBuffer::load_exr(path_albedo, device, oidn_device, command_pool);
    ImageBuffer normal = ImageBuffer::load_exr(path_normal, device, oidn_device, command_pool);

    uint32_t render_width, render_height, target_width, target_height;
    {
        render_width = color.get_width();
        render_height = color.get_height();
        target_width = render_width * scale;
        target_height = render_height * scale;

        std::cout << render_height << " x " << render_width << " -> "
                  << target_height << " x " << target_width << std::endl;
    }

    ImageBuffer out = ImageBuffer(device, oidn_device, target_width, target_height, oidn::Format::Half3);

    // Initialize the denoising filter
    std::cout << "Initializing filter" << std::endl;
    oidn::FilterRef filter = oidn_device.newFilter("RT");


    std::cout << "color in: ";
    std::cout << "    ext mem     " << color.get_buffer().get_ext_mem_type().value() << std::endl;
    std::cout << "    mem handle  " << color.get_buffer().get_ext_mem_handle() << std::endl;
    std::cout << "    size        " << color.get_buffer().get_size() << std::endl;
    std::cout << "    width       " << color.get_width() << std::endl;
    std::cout << "    height      " << color.get_height() << std::endl;
    //std::cout << "format      " << color.get_format() << std::endl;

    std::cout << "out: ";
    std::cout << "    ext mem     " << out.get_buffer().get_ext_mem_type().value() << std::endl;
    std::cout << "    mem handle  " << out.get_buffer().get_ext_mem_handle() << std::endl;
    std::cout << "    size        " << out.get_buffer().get_size() << std::endl;
    std::cout << "    width       " << out.get_width() << std::endl;
    std::cout << "    height      " << out.get_height() << std::endl;

    auto oidn_color = oidn_device.newBuffer(color.get_buffer().get_ext_mem_type()->oidn,
                                            color.get_buffer().get_ext_mem_handle(),
                                            color.get_buffer().get_size());

    auto oidn_albedo = oidn_device.newBuffer(albedo.get_buffer().get_ext_mem_type()->oidn,
                                             albedo.get_buffer().get_ext_mem_handle(),
                                             albedo.get_buffer().get_size());

    auto oidn_normal = oidn_device.newBuffer(normal.get_buffer().get_ext_mem_type()->oidn,
                                             normal.get_buffer().get_ext_mem_handle(),
                                             normal.get_buffer().get_size());

    auto oidn_out = oidn_device.newBuffer(out.get_buffer().get_ext_mem_type()->oidn,
                                          out.get_buffer().get_ext_mem_handle(),
                                          out.get_buffer().get_size());

    filter.setImage("color", oidn_color, color.get_format(), color.get_width(), color.get_height());
    filter.setImage("albedo", oidn_albedo, albedo.get_format(), albedo.get_width(), albedo.get_height());
    filter.setImage("normal", oidn_normal, normal.get_format(), normal.get_width(), normal.get_height());

    filter.setImage("output", oidn_out, out.get_format(), out.get_width(), out.get_height());

    filter.set("hdr", true);
    filter.set("inputScale", scale);
    filter.set("quality", oidn::Quality::Default);

    //const bool showProgress = verbose <= 1;
    //if (showProgress)
    //{
    //  filter.setProgressMonitorFunction(progressCallback);
    //  signal(SIGINT, signalHandler);
    //}

    filter.commit();

    filter.execute();

    out.save_exr("./out.exr", device, command_pool);

    return 0;
}