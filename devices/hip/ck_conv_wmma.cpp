// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_conv.h"
#include "ck_conv.h"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp"

OIDN_NAMESPACE_BEGIN

  template<
    typename T,
    typename AccumT,
    Activation builtinActivation,
    ck::index_t NumGemmKPrefetchStage,
    ck::index_t BlockSize,
    ck::index_t MPerBlock,
    ck::index_t NPerBlock,
    ck::index_t KPerBlock,
    ck::index_t K1,
    ck::index_t MPerWmma,
    ck::index_t NPerWmma,
    ck::index_t MRepeat,
    ck::index_t NRepeat,
    typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
    typename ABlockTransferThreadClusterArrangeOrder,
    typename ABlockTransferSrcAccessOrder,
    ck::index_t ABlockTransferSrcVectorDim,
    ck::index_t ABlockTransferSrcScalarPerVector,
    ck::index_t ABlockTransferDstScalarPerVector_AK1,
    bool ABlockLdsExtraM,
    typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
    typename BBlockTransferThreadClusterArrangeOrder,
    typename BBlockTransferSrcAccessOrder,
    ck::index_t BBlockTransferSrcVectorDim,
    ck::index_t BBlockTransferSrcScalarPerVector,
    ck::index_t BBlockTransferDstScalarPerVector_BK1,
    bool BBlockLdsExtraN,
    ck::index_t CShuffleMRepeatPerShuffle,
    ck::index_t CShuffleNRepeatPerShuffle,
    typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
    ck::index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock>
  class CKConvWMMA final : public Conv
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
      ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<
        2,                       // NDimSpatial
        InLayout,                // ALayout
        WeiLayout,               // BLayout
        ck::Tuple<BiasLayout>,   // DsLayout
        OutLayout,               // ELayout
        InDataType,              // ADataType
        WeiDataType,             // BDataType
        AccDataType,             // AccDataType
        CShuffleDataType,        // CShuffleDataType
        ck::Tuple<BiasDataType>, // DsDataType
        OutDataType,             // EDataType
        InElementOp,             // AElementwiseOperation
        WeiElementOp,            // BElementwiseOperation
        OutElementOp,            // CDEElementwiseOperation
        ConvSpec,                // ConvForwardSpecialization
        GemmSpec,                // GemmSpecialization
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        K1,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
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
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVector_NPerBlock>;

  public:
    static CKConvFactory getFactory()
    {
      return
      {
        [](HIPEngine* engine, const ConvDesc& desc) -> Ref<Conv>
        {
          return makeRef<CKConvWMMA>(engine, desc);
        },
        DataTypeOf<T>::value,
        DataTypeOf<AccumT>::value,
        builtinActivation,
        MPerBlock,
        NPerBlock,
        KPerBlock
      };
    }

    CKConvWMMA(HIPEngine* engine, const ConvDesc& desc)
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
        src ? src->getPtr() : nullptr,        // p_a
        weight ? weight->getPtr() : nullptr,  // p_b
        {bias ? bias->getPtr() : nullptr},    // p_ds
        dst ? dst->getPtr() : nullptr,        // p_e
        getCKTensorLengths(srcDesc),          // a_g_n_c_wis_lengths
        getCKTensorStrides(srcDesc),          // a_g_n_c_wis_strides
        getCKTensorLengths(weightDesc),       // b_g_k_c_xs_lengths
        getCKTensorStrides(weightDesc),       // b_g_k_c_xs_strides
        {getCKTensorLengths(dstDesc)},        // ds_g_n_k_wos_lengths (broadcast to dst shape)
        {getCKTensorStrides(biasDesc)},       // ds_g_n_k_wos_strides
        getCKTensorLengths(dstDesc),          // e_g_n_k_wos_lengths
        getCKTensorStrides(dstDesc),          // e_g_n_k_wos_strides
        {1, 1},                               // conv_filter_strides
        {1, 1},                               // conv_filter_dilations
        {1, 1},                               // input_left_pads
        {1, 1},                               // input_right_pads
        inElementOp,                          // a_element_op
        weiElementOp,                         // b_element_op
        outElementOp);                        // cde_element_op
    }

    DeviceConvFwdInstance conv;
    InElementOp  inElementOp;
    WeiElementOp weiElementOp;
    OutElementOp outElementOp;

    HIPEngine* engine;
  };

  template<Activation activation>
  std::vector<CKConvFactory> getCKConvWMMAInstances()
  {
    return
    {
      // ###############################| Prefetch| Block|  MPer|  NPer|  KPer| K1|  MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
      // ###############################|    Stage|  Size| Block| Block| Block|   |  WMMA| WMMA|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
      // ###############################|         |      |      |      |      |   |      |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
      // ###############################|         |      |      |      |      |   |      |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
      CKConvWMMA<half, float, activation,        1,   128,   256,    32,    32,  8,    16,   16,       8,       1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>::getFactory(),
      CKConvWMMA<half, float, activation,        1,   128,   128,    64,    32,  8,    16,   16,       4,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>::getFactory(),
    };
  }

  template<>
  std::vector<CKConvFactory> getCKConvInstances<HIPArch::WMMA>(Activation activation)
  {
    switch (activation)
    {
    case Activation::None: return getCKConvWMMAInstances<Activation::None>();
    case Activation::ReLU: return getCKConvWMMAInstances<Activation::ReLU>();
    default:               return {};
    }
  }

OIDN_NAMESPACE_END