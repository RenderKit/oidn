// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_conv.h"
#include "ck_conv.h"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp"

OIDN_NAMESPACE_BEGIN

  template<typename T, Activation builtinActivation>
  class CKConvDL final : public Conv
  {
  private:
    using DataType     = typename CKDataType<T>::Type;
    using InDataType   = DataType;
    using WeiDataType  = DataType;
    using BiasDataType = DataType;
    using AccDataType  = float;
    using OutDataType  = DataType;

    using InLayout   = ck::tensor_layout::convolution::NHWGC;
    using WeiLayout  = ck::tensor_layout::convolution::GKYXC;
    using BiasLayout = ck::tensor_layout::convolution::NHWGK;
    using OutLayout  = ck::tensor_layout::convolution::NHWGK;

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

    using DeviceGroupedConvNDFwdInstance =
      ck::tensor_operation::device::DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<
        2,                       // NDimSpatial
        InDataType,              // InDataType
        WeiDataType,             // WeiDataType
        ck::Tuple<BiasDataType>, // MultipleDType
        OutDataType,             // OutDataType
        AccDataType,             // AccDataType
        InLayout,                // InLayout
        WeiLayout,               // WeiLayout
        ck::Tuple<BiasLayout>,   // MultipleDLayout
        OutLayout,               // OutLayout
        InElementOp,             // InElementwiseOperation
        WeiElementOp,            // WeiElementwiseOperation
        OutElementOp,            // OutElementwiseOperation
        ConvSpec,                // ConvolutionForwardSpecialization
        GemmSpec,                // GEMMSpecialization
        256,                     // BlockSize
        128,                     // MPerBlock
        64,                      // NPerBlock
        16,                      // K0PerBlock
        2,                       // K1
        4,                       // M1PerThreadM111
        2,                       // N1PerThreadN111
        1,                       // KPerThread
        S<8, 2>,                 // M11N11ThreadClusterM110Xs
        S<8, 2>,                 // M11N11ThreadClusterN110Xs
        S<8, 1, 1, 2>,           // ABlockTransferThreadSliceLengthsK0_M0_M1_K1
        S<2, 1, 128, 1>,         // ABlockTransferThreadClusterLengthsK0_M0_M1_K1
        S<1, 2, 0, 3>,           // ABlockTransferThreadClusterArrangeOrder
        S<1, 2, 0, 3>,           // ABlockTransferSrcAccessOrder
        S<4, 1, 1, 2>,           // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
        S<1, 2, 0, 3>,           // ABlockTransferSrcVectorTensorContiguousDimOrder
        S<1, 1, 1, 2>,           // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
        S<8, 1, 1, 2>,           // BBlockTransferThreadSliceLengthsK0_N0_N1_K1
        S<2, 1, 64, 1>,          // BBlockTransferThreadClusterLengthsK0_N0_N1_K1
        S<1, 2, 0, 3>,           // BBlockTransferThreadClusterArrangeOrder
        S<1, 2, 0, 3>,           // BBlockTransferSrcAccessOrder
        S<4, 1, 1, 2>,           // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
        S<1, 2, 0, 3>,           // BBlockTransferSrcVectorTensorContiguousDimOrder
        S<1, 1, 1, 2>,           // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
        S<0, 1, 2, 3, 4, 5>,     // CThreadTransferSrcDstAccessOrder
        5,                       // CThreadTransferSrcDstVectorDim
        2                        // CThreadTransferDstScalarPerVector
      >;

  public:
    CKConvDL(HIPEngine* engine, const ConvDesc& desc)
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

    DeviceGroupedConvNDFwdInstance conv;
    InElementOp  inElementOp;
    WeiElementOp weiElementOp;
    OutElementOp outElementOp;

    HIPEngine* engine;
  };

  Ref<Conv> newHIPConvDL(HIPEngine* engine, const ConvDesc& desc)
  {
    if (desc.srcDesc.dataType == DataType::Float16 && desc.activation == Activation::None)
      return makeRef<CKConvDL<half, Activation::None>>(engine, desc);
    if (desc.srcDesc.dataType == DataType::Float16 && desc.activation == Activation::ReLU)
      return makeRef<CKConvDL<half, Activation::ReLU>>(engine, desc);
    throw std::runtime_error("unsupported convolution");
  }

OIDN_NAMESPACE_END