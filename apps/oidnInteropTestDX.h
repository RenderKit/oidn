#include <windows.h>
#include <wrl.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <vector>
#include <optional>
#include <set>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>

#include "OpenImageIO/imageio.h"

#include "OpenImageDenoise/oidn.hpp"

using Microsoft::WRL::ComPtr;

class CmdList {
public:
    CmdList(ComPtr<ID3D12Device> device) {
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)))) {
            throw std::runtime_error("Failed to create command allocator.");
        }

        if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator.Get(), nullptr, IID_PPV_ARGS(&commandList)))) {
            throw std::runtime_error("Failed to create command list.");
        }
    }

    CmdList(const CmdList& other) = delete;
    CmdList& operator=(const CmdList& other) = delete;
    CmdList(CmdList&& other) noexcept : commandList(std::move(other.commandList)) {}
    CmdList& operator=(CmdList&& other) noexcept {
        if (this != &other) {
            commandList = std::move(other.commandList);
        }
        return *this;
    }

    ~CmdList() {
        if (commandList) {
            std::cout << "Releasing command list." << std::endl;
            commandList.Reset();
            commandAllocator.Reset();
        }
    }

public:
    ID3D12GraphicsCommandList* operator->() { return commandList.Get(); }
    ID3D12GraphicsCommandList& operator*() { return *commandList.Get(); }

    void closeAndWait(ComPtr<ID3D12Device>& device, ComPtr<ID3D12CommandQueue>& commandQueue) {
        // Close and execute command list
        if (FAILED(commandList->Close())) {
            throw std::runtime_error("Failed to close command list.");
        }
        ID3D12CommandList* ppCommandLists[] = { commandList.Get() };
        commandQueue->ExecuteCommandLists(1, ppCommandLists);

        // Create fence for synchronization
        ComPtr<ID3D12Fence> fence;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
            throw std::runtime_error("Failed to create fence.");
        }
        HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!fenceEvent) {
            throw std::runtime_error("Failed to create fence event.");
        }
        UINT64 fenceValue = 1;
        if (FAILED(commandQueue->Signal(fence.Get(), fenceValue))) {
            CloseHandle(fenceEvent);
            throw std::runtime_error("Failed to signal fence.");
        }
        if (fence->GetCompletedValue() < fenceValue) {
            if (FAILED(fence->SetEventOnCompletion(fenceValue, fenceEvent))) {
                CloseHandle(fenceEvent);
                throw std::runtime_error("Failed to set event on completion.");
            }
            WaitForSingleObject(fenceEvent, INFINITE);
        }
        CloseHandle(fenceEvent);
    }

private:
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> commandList;
};


class Buffer {
public:
    Buffer(ComPtr<ID3D12Device> device, size_t size, D3D12_HEAP_TYPE heapType, D3D12_HEAP_FLAGS heapFlags, D3D12_RESOURCE_FLAGS resourceFlags, D3D12_RESOURCE_STATES state): size(size) {
        // Describe the buffer
        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Alignment = 0;
        bufferDesc.Width = size;
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.SampleDesc.Quality = 0;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.Flags = resourceFlags;

        // Heap properties for shared resource
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = heapType;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapProps.CreationNodeMask = 1;
        heapProps.VisibleNodeMask = 1;

        // Create the buffer resource
        HRESULT hr = device->CreateCommittedResource(
            &heapProps,
            heapFlags,
            &bufferDesc,
            state,
            nullptr,
            IID_PPV_ARGS(&buffer)
        );
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to create buffer resource.");
        }
        std::cout << "Buffer resource created successfully." << std::endl;
    }

    static Buffer make_image_buffer(ComPtr<ID3D12Device>& device, uint32_t width, uint32_t height, oidn::Format format) {
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

        return Buffer(device, size_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); 
    }

    static Buffer make_staging_buffer(ComPtr<ID3D12Device> device, size_t size, char rw) {
        D3D12_HEAP_TYPE heapType;
        D3D12_RESOURCE_STATES state;
        if (rw == 'w') {
            heapType = D3D12_HEAP_TYPE_UPLOAD;
            state = D3D12_RESOURCE_STATE_GENERIC_READ;
        } else if (rw == 'r') {
            heapType = D3D12_HEAP_TYPE_READBACK;
            state = D3D12_RESOURCE_STATE_COPY_DEST;
        } else {
            throw std::invalid_argument("rw must be 'r' or 'w'");
        }

        return Buffer(device, size, heapType, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_FLAG_NONE, state);
    }

    Buffer(const Buffer& other) = delete;
    Buffer& operator=(const Buffer& other) = delete;
    Buffer(Buffer&& other) noexcept
        : size(other.size), buffer(std::move(other.buffer))
    {
        other.size = 0;
        other.buffer = nullptr;
    }
    Buffer& operator=(Buffer&& other) noexcept
    {
        if (this != &other)
        {
            size = other.size;
            buffer = std::move(other.buffer);
            other.size = 0;
            other.buffer = nullptr;
        }
        return *this;
    }
    ~Buffer() {
        if (buffer) {
            std::cout << "Releasing buffer resource." << std::endl;
            buffer.Reset();
        }
    }


public:
    operator ComPtr<ID3D12Resource>() const { return get(); }
    ComPtr<ID3D12Resource> get() const { return buffer; }
    size_t get_size() const { return size; }

private:
    size_t size;
    ComPtr<ID3D12Resource> buffer = nullptr;
};



class ImageBuffer {
public:
    ImageBuffer(ComPtr<ID3D12Device>& device, uint32_t width, uint32_t height, oidn::Format format)
        : buffer(Buffer::make_image_buffer(device, width, height, format)),
          width(width), height(height), format(format) {}

    ImageBuffer(const ImageBuffer&) = delete;
    ImageBuffer& operator=(const ImageBuffer&) = delete;
    ImageBuffer(ImageBuffer&&) = default;
    ImageBuffer& operator=(ImageBuffer&&) = default;

    ~ImageBuffer() = default;


public:
    static ImageBuffer load_exr(std::string filename, ComPtr<ID3D12Device>& device, ComPtr<ID3D12CommandQueue>& commandQueue) {
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


        ImageBuffer buffer(device, width, height, format);
        Buffer stagingBuffer = Buffer::make_staging_buffer(device, buffer.buffer.get_size(), 'w');

        auto stagingResource = stagingBuffer.get();

        void* mapped = nullptr;
        stagingResource->Map(0, nullptr, &mapped);
        in->read_image(0, 0, 0, nchannels, spec.channelformat(0), mapped);
        D3D12_RANGE readingRange = {0, buffer.buffer.get_size()};
        stagingResource->Unmap(0, &readingRange);
        in->close();

        CmdList cl(device);
        // Transition buffer to COPY_DEST
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = buffer.get_buffer().get().Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cl->ResourceBarrier(1, &barrier);

        // Copy data from staging buffer to buffer
        cl->CopyResource(buffer.get_buffer().get().Get(), stagingBuffer.get().Get());

        // Transition buffer back to UNORDERED_ACCESS
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        cl->ResourceBarrier(1, &barrier);

        cl.closeAndWait(device, commandQueue);

        return buffer;
    }

    void save_exr(std::string filename, ComPtr<ID3D12Device>& device, ComPtr<ID3D12CommandQueue>& commandQueue) {
        // Create a staging buffer to copy the data to
        Buffer stagingBuffer = Buffer::make_staging_buffer(device, buffer.get_size(), 'r');

        // Create command list for copy operation
        CmdList cl(device);

        // Transition buffer to COPY_SOURCE
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = buffer.get().Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cl->ResourceBarrier(1, &barrier);

        // Copy buffer to staging
        cl->CopyResource(stagingBuffer.get().Get(), buffer.get().Get());

        // Transition buffer back to UNORDERED_ACCESS
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        cl->ResourceBarrier(1, &barrier);

        cl.closeAndWait(device, commandQueue);


        // create output file
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

        void* mapped = nullptr;
        auto stagingResource = stagingBuffer.get();
        stagingResource->Map(0, nullptr, &mapped);
        out->write_image(type, mapped);
        D3D12_RANGE readingRange = {0, buffer.get_size()};
        stagingResource->Unmap(0, &readingRange);
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



class InteropTestDX {
public:
    InteropTestDX() {
#if defined(_DEBUG)
        {
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
                debugController->EnableDebugLayer();
            }
        }
#endif

        // Create DXGI factory
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
            throw std::runtime_error("Failed to create DXGI Factory.");
        }

        // Create D3D12 device
        if (FAILED(D3D12CreateDevice(
            nullptr, // default adapter
            D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&device)
        ))) {
            throw std::runtime_error("Failed to create D3D12 Device.");
        }

        // Create command queue
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        if (FAILED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)))) {
            throw std::runtime_error("Failed to create command queue.");
        }
    }

    ~InteropTestDX() = default;

    InteropTestDX(const InteropTestDX&) = delete;
    InteropTestDX& operator=(const InteropTestDX&) = delete;
    InteropTestDX(InteropTestDX&&) = default;
    InteropTestDX& operator=(InteropTestDX&&) = default;



    void setup_oidn_device() {
        // Get the LUID of the adapter
        LUID luid = device->GetAdapterLuid();

        // Initialize the denoiser device
        oidn_device = oidn::newDevice(oidn::LUID{luid.LowPart, luid.HighPart});
        if (oidn_device.getError() != oidn::Error::None)
            throw std::runtime_error("Failed to create OIDN device.");
        oidn_device.commit();
        if (oidn_device.getError() != oidn::Error::None)
            throw std::runtime_error("Failed to commit OIDN device.");

        // Find a compatible external memory handle type
        const auto oidn_external_mem_types = oidn_device.get<oidn::ExternalMemoryTypeFlags>("externalMemoryTypes");
        if (!(oidn_external_mem_types & oidn::ExternalMemoryTypeFlag::OpaqueWin32))
            throw std::runtime_error("failed to find compatible external memory type");
    }

    ImageBuffer create_buffer(uint32_t width, uint32_t height, oidn::Format format) {
        return ImageBuffer(device, width, height, format);
    }

    ImageBuffer load_exr(std::string& path) {
        return ImageBuffer::load_exr(path, device, commandQueue);
    }

    oidn::BufferRef create_oidn_buffer(ImageBuffer& imageBuffer) {
        // Create an OIDN buffer from the D3D12 resource
        HANDLE sharedHandle = nullptr;
        device->CreateSharedHandle(imageBuffer.get_buffer().get().Get(), nullptr, GENERIC_ALL, nullptr, &sharedHandle);
        return oidn_device.newBuffer(oidn::ExternalMemoryTypeFlag::OpaqueWin32, &sharedHandle, nullptr, imageBuffer.get_buffer().get_size());
    }

    void save_exr(ImageBuffer& buf, const std::string& path) {
        return buf.save_exr(path, device, commandQueue);
    }

public:
    ComPtr<ID3D12Debug> debugController;
    ComPtr<IDXGIFactory6> factory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> commandQueue;
    oidn::DeviceRef oidn_device;
};