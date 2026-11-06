// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_conv.h"

// This translation unit is compiled only for gfx950 (wave64). ROCm 7's warpSize
// is not a constant expression, so Composable Kernel's ck::get_warp_size() cannot
// be used from constexpr code. The variable is declared by the HIP headers; the
// macro then makes ck::get_warp_size() a constexpr 64. RDNA sources do not do this.
#include <hip/hip_runtime.h>
#define warpSize 64

#include "ck_conv.h"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"

OIDN_NAMESPACE_BEGIN

  // CDNA4 (gfx950) convolution. The tile is the double-rate MFMA instance from
  // Composable Kernel's grouped-conv example, which is the one marked for gfx950.
  template<
    typename T,
    typename AccumT,
    Activation builtinActivation,
    ck::index_t NumGemmKPrefetchStage,
    ck::index_t BlockSize,
    ck::index_t MPerBlock,
    ck::index_t NPerBlock,
    ck::index_t KPerBlock,
    ck::index_t AK1,
    ck::index_t BK1,
    ck::index_t MPerXDL,
    ck::index_t NPerXDL,
    ck::index_t MXdlPerWave,
    ck::index_t NXdlPerWave,
    typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
    typename ABlockTransferThreadClusterArrangeOrder,
    typename ABlockTransferSrcAccessOrder,
    ck::index_t ABlockTransferSrcVectorDim,
    ck::index_t ABlockTransferSrcScalarPerVector,
    ck::index_t ABlockTransferDstScalarPerVector_AK1,
    ck::index_t ABlockLdsExtraM,
    typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
    typename BBlockTransferThreadClusterArrangeOrder,
    typename BBlockTransferSrcAccessOrder,
    ck::index_t BBlockTransferSrcVectorDim,
    ck::index_t BBlockTransferSrcScalarPerVector,
    ck::index_t BBlockTransferDstScalarPerVector_BK1,
    ck::index_t BBlockLdsExtraN,
    ck::index_t CShuffleMXdlPerWavePerShuffle,
    ck::index_t CShuffleNXdlPerWavePerShuffle,
    typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
    ck::index_t CDEBlockTransferScalarPerVector_NPerBlock>
  class CKConvMFMA final : public Conv
  {
  private:
    using DataType         = typename CKDataType<T>::Type;
    using InDataType       = DataType;
    using WeiDataType      = DataType;
    using BiasDataType     = DataType;
    using AccDataType      = typename CKDataType<AccumT>::Type;
    using CShuffleDataType = DataType;
    using OutDataType      = DataType;

    using InLayout   = ck::tensor_layout::convolution::G_NHW_C;
    using WeiLayout  = ck::tensor_layout::convolution::G_K_YX_C;
    using BiasLayout = ck::tensor_layout::convolution::G_NHW_K;
    using OutLayout  = ck::tensor_layout::convolution::G_NHW_K;

    using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
    using OutElementOp = std::conditional_t<builtinActivation == Activation::ReLU,
                                            ck::tensor_operation::element_wise::AddRelu,
                                            ck::tensor_operation::element_wise::Add>;

    static constexpr auto ConvSpec =
      ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

    static constexpr auto GemmSpec =
      ck::tensor_operation::device::GemmSpecialization::MNKPadding;

    using DeviceConvFwdInstance =
      ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        2,
        InLayout,
        WeiLayout,
        ck::Tuple<BiasLayout>,
        OutLayout,
        InDataType,
        WeiDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<BiasDataType>,
        OutDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvSpec,
        GemmSpec,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock>;

  public:
    static CKConvFactory getFactory()
    {
      return
      {
        [](HIPEngine* engine, const ConvDesc& desc) -> Ref<Conv>
        {
          return makeRef<CKConvMFMA>(engine, desc);
        },
        DataTypeOf<T>::value,
        DataTypeOf<AccumT>::value,
        builtinActivation,
        MPerBlock,
        NPerBlock,
        KPerBlock
      };
    }

    CKConvMFMA(HIPEngine* engine, const ConvDesc& desc)
      : Conv(desc),
        engine(engine)
    {
      if (srcDesc.dataType != DataTypeOf<T>::value)
        throw std::invalid_argument("unexpected convolution source data type");
      if (weightDesc.dataType != srcDesc.dataType || biasDesc.dataType != srcDesc.dataType)
        throw std::invalid_argument("unsupported convolution weight/bias data type");
      if (activation != builtinActivation)
        throw std::invalid_argument("unexpected convolution activation function");
    }

    Engine* getEngine() const override { return engine; }

    bool isSupported() const override
    {
      return conv.IsSupportedArgument(makeArgument());
    }

    void submitKernels(const Ref<CancellationToken>& ct) override
    {
      if (!src || !weight || !bias || !dst)
        throw std::logic_error("convolution argument not set");

      auto invoker = conv.MakeInvoker();
      auto argument = makeArgument();
      invoker.Run(argument, StreamConfig{});
    }

  private:
    auto makeArgument() const
    {
      return conv.MakeArgument(
        src ? src->getPtr() : nullptr,
        weight ? weight->getPtr() : nullptr,
        {bias ? bias->getPtr() : nullptr},
        dst ? dst->getPtr() : nullptr,
        getCKTensorLengths(srcDesc),
        getCKTensorStrides(srcDesc),
        getCKTensorLengths(weightDesc),
        getCKTensorStrides(weightDesc),
        {getCKTensorLengths(dstDesc)},
        {getCKTensorStrides(biasDesc)},
        getCKTensorLengths(dstDesc),
        getCKTensorStrides(dstDesc),
        {1, 1},
        {1, 1},
        {1, 1},
        {1, 1},
        inElementOp,
        weiElementOp,
        outElementOp);
    }

    DeviceConvFwdInstance conv;
    InElementOp  inElementOp;
    WeiElementOp weiElementOp;
    OutElementOp outElementOp;

    HIPEngine* engine;
  };

  template<Activation activation>
  std::vector<CKConvFactory> getCKConvMFMAInstances()
  {
    return
    {
      CKConvMFMA<half, float, activation,
                 1, 256, 128, 64, 64, 16, 16, 32, 32, 2, 1,
                 S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 4, 4, 1,
                 S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 4, 4, 1,
                 1, 1, S<1, 16, 1, 16>, 4>::getFactory(),
    };
  }

  template<>
  std::vector<CKConvFactory> getCKConvInstances<HIPArch::MFMA>(Activation activation)
  {
    switch (activation)
    {
    case Activation::None: return getCKConvMFMAInstances<Activation::None>();
    case Activation::ReLU: return getCKConvMFMAInstances<Activation::ReLU>();
    default:               return {};
    }
  }

OIDN_NAMESPACE_END
