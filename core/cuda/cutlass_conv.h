// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "../tensor.h"
#include "cuda_conv.h"

namespace oidn {

  template<> struct DataTypeOf<cutlass::half_t> { static constexpr DataType value = DataType::Float16; };

  template<typename T>
  struct CutlassElement { using Type = T; };

  template<>
  struct CutlassElement<half> { using Type = cutlass::half_t; };

  template<typename Element, typename SmArch>
  struct CutlassMathInstruction;

  template<>
  struct CutlassMathInstruction<cutlass::half_t, cutlass::arch::Sm80>
  {
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    static constexpr int alignment = 8;
  };

  template<>
  struct CutlassMathInstruction<cutlass::half_t, cutlass::arch::Sm75>
  {
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    static constexpr int alignment = 8;
  };

  template<>
  struct CutlassMathInstruction<cutlass::half_t, cutlass::arch::Sm70>
  {
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
    static constexpr int alignment = 8;
  };

  template<>
  struct CutlassMathInstruction<cutlass::half_t, cutlass::arch::Sm60>
  {
    using MMAOp = cutlass::arch::OpClassSimt;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    static constexpr int alignment = 1;
  };

  template<typename Element, int alignment>
  struct CutlassEpilogueTraits
  {
    static constexpr int elementBits  = cutlass::sizeof_bits<Element>::value;
    static constexpr int alignmentC   = min(alignment, 8);
    static constexpr int vectorLength = min(alignmentC * elementBits, 128) / elementBits;
  };

  template<typename Element, Activation, int alignment>
  struct CutlassEpilogue;

  template<typename Element, int alignment>
  struct CutlassEpilogue<Element, Activation::None, alignment>
  {
    using Op = cutlass::epilogue::thread::LinearCombination<
      Element, // ElementOutput
      CutlassEpilogueTraits<Element, alignment>::vectorLength,
      Element, // ElementAccumulator
      Element, // ElementCompute
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // alpha * C + D
  };

  template<typename Element, int alignment>
  struct CutlassEpilogue<Element, Activation::ReLU, alignment>
  {
    using Op = cutlass::epilogue::thread::LinearCombinationRelu<
      Element, // ElementOutput
      CutlassEpilogueTraits<Element, alignment>::vectorLength,
      Element, // ElementAccumulator
      Element, // ElementCompute
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // alpha * C + D
  };

  inline void checkError(cutlass::Status status)
  {
    if (status != cutlass::Status::kSuccess)
      throw Exception(Error::Unknown, "CUTLASS error");
  }

  inline cutlass::Tensor4DCoord toCutlassTensor4DCoord(const TensorDesc& td)
  {
    switch (td.layout)
    {
    case TensorLayout::hwc:
      return {1, td.getH(), td.getW(), td.getC()};
    case TensorLayout::ohwi:
      return {td.getO(), td.getH(), td.getW(), td.getI()};
    default:
      throw std::invalid_argument("unsupported tensor layout");
    }
  }

  template<typename T>
  cutlass::TensorRef<T, cutlass::layout::TensorNHWC> toCutlassTensorRef(const TensorDesc& td)
  {
    if (td.dataType != DataTypeOf<T>::value)
      throw std::logic_error("tensor data type mismatch");

    switch (td.layout)
    {
    case TensorLayout::x:
      return {nullptr, cutlass::layout::TensorNHWC::Stride(0)};
    case TensorLayout::hwc:
    case TensorLayout::ohwi:
      return {nullptr, cutlass::layout::TensorNHWC::packed(toCutlassTensor4DCoord(td))};
    default:
      throw std::invalid_argument("unsupported tensor layout");
    }
  }

  template<typename T>
  cutlass::TensorRef<T, cutlass::layout::TensorNHWC> toCutlassTensorRef(const std::shared_ptr<Tensor>& t)
  {
    if (t->getDataType() != DataTypeOf<T>::value)
      throw std::logic_error("tensor data type mismatch");

    switch (t->getLayout())
    {
    case TensorLayout::x:
      return {(T*)t->getData(), cutlass::layout::TensorNHWC::Stride(0)};
    case TensorLayout::hwc:
    case TensorLayout::ohwi:
      return {(T*)t->getData(), cutlass::layout::TensorNHWC::packed(toCutlassTensor4DCoord(t->getDesc()))};
    default:
      throw std::invalid_argument("unsupported tensor layout");
    }
  }

  inline cutlass::conv::Conv2dProblemSize toCutlassProblemSize(const ConvDesc& desc)
  {
    return {
      toCutlassTensor4DCoord(desc.srcDesc),
      toCutlassTensor4DCoord(desc.weightDesc),
      {1, 1, 1, 1}, // padding
      {1, 1},       // stride
      {1, 1},       // dilation
      cutlass::conv::Mode::kCrossCorrelation,
      1 // split-k slices
    };
  }

  template<
    typename T,
    typename SmArch,
    typename ThreadblockShape,
    typename WarpShape,
    int numStages,
    Activation builtinActivation>
  class CutlassConv final : public Conv
  {
  private:
    using Element = typename CutlassElement<T>::Type;
    using ElementAccumulator = Element;
    using ElementComputeEpilogue = ElementAccumulator;
    using ElementInputA = Element;
    using ElementInputB = Element;
    using ElementOutput = Element;

    using LayoutInputA = cutlass::layout::TensorNHWC;
    using LayoutInputB = cutlass::layout::TensorNHWC;
    using LayoutOutput = cutlass::layout::TensorNHWC;

    using MathInstruction = CutlassMathInstruction<Element, SmArch>;
    using MMAOp = typename MathInstruction::MMAOp;
    using InstructionShape = typename CutlassMathInstruction<Element, SmArch>::InstructionShape;
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    using EpilogueOp = typename CutlassEpilogue<Element, builtinActivation, MathInstruction::alignment>::Op;

    using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
      ElementInputA, LayoutInputA,
      ElementInputB, LayoutInputB,
      ElementOutput, LayoutOutput,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      numStages,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized
    >::Kernel;

    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  public:
    CutlassConv(const Ref<CUDADevice>& device, const ConvDesc& desc)
      : Conv(desc),
        device(device)
    {
      if (activation != builtinActivation)
        throw std::logic_error("incompatible convolution activation function");
      if (weightDesc.dataType != srcDesc.dataType || biasDesc.dataType != srcDesc.dataType)
        throw std::invalid_argument("unsupported combination of convolution argument data types");

      problemSize = toCutlassProblemSize(desc);

      initialArguments = {
        problemSize,
        toCutlassTensorRef<ElementInputA>(srcDesc),
        toCutlassTensorRef<ElementInputB>(weightDesc),
        toCutlassTensorRef<ElementOutput>(biasDesc),
        toCutlassTensorRef<ElementOutput>(dstDesc),
        {ElementComputeEpilogue(1)}
      };
    }

    bool isSupported() const override
    {
      return gemm.can_implement(initialArguments) == cutlass::Status::kSuccess;
    }

    size_t getScratchByteSize() const override
    {
      assert(isSupported());
      return gemm.get_workspace_size(initialArguments);
    }

    void setScratch(const std::shared_ptr<Tensor>& scratch) override
    {
      // FIXME: check size
      this->scratch = scratch;
    }

    void finalize() override
    {
      checkError(gemm.initialize(initialArguments, scratch ? scratch->getData() : nullptr));
      finalized = true;
    }

    void run() override
    {
      assert(isSupported());
      if (!finalized)
        throw std::logic_error("convolution not finalized");
      if (!src || !weight || !bias || !dst)
        throw std::logic_error("convolution argument not set");

      typename ImplicitGemm::Arguments arguments {
        problemSize,
        toCutlassTensorRef<ElementInputA>(src),
        toCutlassTensorRef<ElementInputB>(weight),
        toCutlassTensorRef<ElementOutput>(bias),
        toCutlassTensorRef<ElementOutput>(dst),
        {ElementComputeEpilogue(1)}
      };

      checkError(gemm.update(arguments, scratch ? scratch->getData() : nullptr));
      checkError(gemm());
    }

  private:
    Ref<CUDADevice> device;
    bool finalized = false;
    cutlass::conv::Conv2dProblemSize problemSize;
    typename ImplicitGemm::Arguments initialArguments;
    ImplicitGemm gemm;
    std::shared_ptr<Tensor> scratch;
  };

  struct CutlassConvFactory
  {
    std::shared_ptr<Conv> (*make)(const Ref<CUDADevice>&, const ConvDesc&);

    DataType dataType;
    int sm;                     // compute capability
    int blockM, blockN, blockK; // threadblock size
  };

  template<
    typename T,
    typename SmArch,
    typename ThreadblockShape,
    typename WarpShape,
    int numStages>
  class CutlassConvInstance
  {
  public:
    static CutlassConvFactory get()
    {
      return {
        make,
        DataTypeOf<T>::value,
        SmArch::kMinComputeCapability,
        ThreadblockShape::kM,
        ThreadblockShape::kN,
        ThreadblockShape::kK
      };
    }

  private:
    template<Activation activation>
    using CutlassConvType = CutlassConv<T, SmArch, ThreadblockShape, WarpShape, numStages, activation>;

    static std::shared_ptr<Conv> make(const Ref<CUDADevice>& device, const ConvDesc& desc)
    {
      switch (desc.activation)
      {
      case Activation::None:
        return std::make_shared<CutlassConvType<Activation::None>>(device, desc);
      case Activation::ReLU:
        return std::make_shared<CutlassConvType<Activation::ReLU>>(device, desc);
      default:
        throw std::invalid_argument("unsupported convolution activation function");
      }
    }
  };

  template<int sm>
  std::vector<CutlassConvFactory> getCutlassConvInstances();

} // namespace oidn