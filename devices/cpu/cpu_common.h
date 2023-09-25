// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/image.h"
#include "core/tensor.h"
#include "core/tile.h"
#include "core/color.h"
#include "cpu_input_process_ispc.h"
#include "cpu_convolution_ispc.h"
#include "color_ispc.h"

OIDN_NAMESPACE_BEGIN

  ispc::ImageAccessor toISPC(Image& image);

  template<typename T> T toISPC(Tensor& tensor);

  template<> ispc::TensorAccessor3D toISPC<ispc::TensorAccessor3D>(Tensor& tensor);

#if defined(OIDN_ISPC)
  template<> ispc::TensorAccessor1D toISPC<ispc::TensorAccessor1D>(Tensor& tensor);
  template<> ispc::TensorAccessor4D toISPC(Tensor& tensor);
#endif

  ispc::Tile toISPC(const Tile& tile);
  ispc::TransferFunction toISPC(const TransferFunction& tf);

OIDN_NAMESPACE_END