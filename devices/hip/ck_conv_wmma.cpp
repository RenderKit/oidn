// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_conv.h"
#include "ck_conv.h"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp"

OIDN_NAMESPACE_BEGIN

  template<typename T, Activation builtinActivation>
  class CKConvWMMA final : public Conv
  {
  private:
    using DataType         = typename CKDataType<T>::Type;
    using InDataType       = DataType;
    using WeiDataType      = DataType;
    using BiasDataType     = DataType;
    using AccDataType      = float;
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

    template<ck::index_t... Is>
    using S = ck::Sequence<Is...>;

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
        1,                       // PrefetchStage
        64,                      // BlockSize
        64,                      // MPerBlock
        32,                      // NPerBlock
        32,                      // KPerBlock
        8,                       // K1
        16,                      // MPerWMMA
        16,                      // NPerWMMA
        2,                       // MRepeat
        2,                       // NRepeat
        S<4, 16, 1>,             // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,              // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,              // ABlockTransferSrcAccessOrder
        2,                       // ABlockTransferSrcVectorDim
        8,                       // ABlockTransferSrcScalarPerVector
        8,                       // ABlockTransferDstScalarPerVector_K1
        1,                       // ABlockLdsExtraM
        S<4, 16, 1>,             // BBlockTransferThreadClusterLengths_BK0_N_K1
        S<1, 0, 2>,              // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,              // BBlockTransferSrcAccessOrder
        2,                       // BBlockTransferSrcVectorDim
        8,                       // BBlockTransferSrcScalarPerVector
        8,                       // BBlockTransferDstScalarPerVector_K1
        1,                       // BBlockLdsExtraN
        1,                       // CShuffleMXdlPerWavePerShuffle
        1,                       // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 2>,          // CDEShuffleBlockTransferClusterLengths_MBlock_MWaveMPerXdl_NBlock_NWaveNPerXdl
        8                        // CDEShuffleBlockTransferScalarPerVector_NPerBlock
      >;

  public:
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

  Ref<Conv> newHIPConvWMMA(HIPEngine* engine, const ConvDesc& desc)
  {
    if (desc.srcDesc.dataType == DataType::Float16 && desc.activation == Activation::None)
      return makeRef<CKConvWMMA<half, Activation::None>>(engine, desc);
    if (desc.srcDesc.dataType == DataType::Float16 && desc.activation == Activation::ReLU)
      return makeRef<CKConvWMMA<half, Activation::ReLU>>(engine, desc);
    throw std::runtime_error("unsupported convolution");
  }

OIDN_NAMESPACE_END