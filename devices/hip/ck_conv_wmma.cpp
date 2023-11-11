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
        ck::Tuple<BiasDataType>, // DsDataType
        OutDataType,             // EDataType
        AccDataType,             // AccDataType
        CShuffleDataType,        // CShuffleDataType
        InElementOp,             // AElementwiseOperation
        WeiElementOp,            // BElementwiseOperation
        OutElementOp,            // CDEElementwiseOperation
        ConvSpec,                // ConvForwardSpecialization
        GemmSpec,                // GemmSpecialization
        256,                     // BlockSize
        128,                     // MPerBlock
        64,                      // NPerBlock
        4,                       // K0PerBlock
        8,                       // K1
        16,                      // MPerWMMA
        16,                      // NPerWMMA
        4,                       // MRepeat
        1,                       // NRepeat
        S<4, 64, 1>,             // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,              // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,              // ABlockTransferSrcAccessOrder
        2,                       // ABlockTransferSrcVectorDim
        8,                       // ABlockTransferSrcScalarPerVector
        8,                       // ABlockTransferDstScalarPerVector_AK1
        true,                    // ABlockLdsExtraM
        S<4, 64, 1>,             // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,              // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,              // BBlockTransferSrcAccessOrder
        2,                       // BBlockTransferSrcVectorDim
        8,                       // BBlockTransferSrcScalarPerVector
        8,                       // BBlockTransferDstScalarPerVector_BK1
        true,                    // BBlockLdsExtraN
        1,                       // CShuffleMRepeatPerShuffle
        1,                       // CShuffleNRepeatPerShuffle
        S<1, 32, 1, 8>,          // CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8                        // CDEShuffleBlockTransferScalarPerVector_NPerBlock
      >;

  public:
    CKConvWMMA(const Ref<HIPEngine>& engine, const ConvDesc& desc)
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

    bool isSupported() const override
    {
      return conv.IsSupportedArgument(makeArgument());
    }

    void submit() override
    {
      if (!src || !weight || !bias || !dst)
        throw std::logic_error("convolution argument not set");

      auto invoker = conv.MakeInvoker();
      auto argument = makeArgument();
      invoker.Run(argument, StreamConfig{});
    }

  private:
    void updateWeight() override
    {
      weight = weight->toDevice(engine);
    }

    void updateBias() override
    {
      bias = bias->toDevice(engine);
    }

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

    Ref<HIPEngine> engine;
  };

  std::shared_ptr<Conv> newHIPConvWMMA(const Ref<HIPEngine>& engine, const ConvDesc& desc)
  {
    if (desc.srcDesc.dataType == DataType::Float16 && desc.activation == Activation::None)
      return std::make_shared<CKConvWMMA<half, Activation::None>>(engine, desc);
    if (desc.srcDesc.dataType == DataType::Float16 && desc.activation == Activation::ReLU)
      return std::make_shared<CKConvWMMA<half, Activation::ReLU>>(engine, desc);
    throw std::runtime_error("unsupported convolution");
  }

OIDN_NAMESPACE_END