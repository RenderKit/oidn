// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_conv.h"
#include "ck_conv.h"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp"

OIDN_NAMESPACE_BEGIN

  template<
    typename T,
    typename AccumT,
    Activation builtinActivation,
    ck::index_t BlockSize,
    ck::index_t MPerBlock,
    ck::index_t NPerBlock,
    ck::index_t K0PerBlock,
    ck::index_t K1,
    ck::index_t M1PerThread,
    ck::index_t N1PerThread,
    ck::index_t KPerThread,
    typename M1N1ThreadClusterM1Xs,
    typename M1N1ThreadClusterN1Xs,
    typename ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
    typename ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
    typename ABlockTransferThreadClusterArrangeOrder,
    typename ABlockTransferSrcAccessOrder,
    typename ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
    typename ABlockTransferSrcVectorTensorContiguousDimOrder,
    typename ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
    typename BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
    typename BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
    typename BBlockTransferThreadClusterArrangeOrder,
    typename BBlockTransferSrcAccessOrder,
    typename BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
    typename BBlockTransferSrcVectorTensorContiguousDimOrder,
    typename BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
    typename CThreadTransferSrcDstAccessOrder,
    ck::index_t CThreadTransferSrcDstVectorDim,
    ck::index_t CThreadTransferDstScalarPerVector>
  class CKConvDL final : public Conv
  {
  private:
    using DataType     = typename CKDataType<T>::Type;
    using InDataType   = DataType;
    using WeiDataType  = DataType;
    using BiasDataType = DataType;
    using AccDataType  = typename CKDataType<AccumT>::Type;
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
        BlockSize,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        K1,
        M1PerThread,
        N1PerThread,
        KPerThread,
        M1N1ThreadClusterM1Xs,
        M1N1ThreadClusterN1Xs,
        ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
        ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
        ABlockTransferSrcVectorTensorContiguousDimOrder,
        ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
        BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
        BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
        BBlockTransferSrcVectorTensorContiguousDimOrder,
        BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
        CThreadTransferSrcDstAccessOrder,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector>;

  public:
    static CKConvFactory getFactory()
    {
      return
      {
        [](HIPEngine* engine, const ConvDesc& desc) -> Ref<Conv>
        {
          return makeRef<CKConvDL>(engine, desc);
        },
        DataTypeOf<T>::value,
        DataTypeOf<AccumT>::value,
        builtinActivation,
        MPerBlock,
        NPerBlock,
        K0PerBlock
      };
    }

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

    DeviceGroupedConvNDFwdInstance conv;
    InElementOp  inElementOp;
    WeiElementOp weiElementOp;
    OutElementOp outElementOp;

    HIPEngine* engine;
  };

  template<Activation activation>
  std::vector<CKConvFactory> getCKConvDLInstances()
  {
    return
    {
      // #############################| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
      // #############################|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
      // #############################|      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
      // #############################|      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
      CKConvDL<half, float, activation,   256,   128,   128,    16,  2,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 2>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,       S<1, 1, 1, 2>,      S<8, 1, 1, 2>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,       S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>,               5,                  4>::getFactory(),
    };
  }

  template<>
  std::vector<CKConvFactory> getCKConvInstances<HIPArch::DL>(Activation activation)
  {
    switch (activation)
    {
    case Activation::None: return getCKConvDLInstances<Activation::None>();
    case Activation::ReLU: return getCKConvDLInstances<Activation::ReLU>();
    default:               return {};
    }
  }

OIDN_NAMESPACE_END