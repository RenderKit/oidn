// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal_kernel_constants.h"

template<typename TensorDataT>
class TransferFunction
{
public:
  TransferFunction(TransferFunctionType type, float normScale)
    : type(type), normScale(normScale)
  {
    rcpNormScale = 1.f / normScale;
  }
  
  TensorDataT forward(TensorDataT y)
  {
    switch(type)
    {
      case TransferFunctionType::PU:
        return forward_pu(y) * normScale;
      case TransferFunctionType::Linear:
        return forward_linear(y);
      case TransferFunctionType::Log:
        return forward_log(y) * normScale;
      case TransferFunctionType::SRGB:
        return forward_srgb(y);
    }
  }
  
  TensorDataT inverse(TensorDataT y)
  {
    switch(type)
    {
      case TransferFunctionType::PU:
        return inverse_pu(y * rcpNormScale);
      case TransferFunctionType::Linear:
        return inverse_linear(y);
      case TransferFunctionType::Log:
        return inverse_log(y * rcpNormScale);
      case TransferFunctionType::SRGB:
        return inverse_srgb(y);
    }
  }
  
private:
  TensorDataT forward_pu(TensorDataT y)
  {
    if (y <= PU_Y0)
      return (PU_A * y);
    else if (y <= PU_Y1)
      return (PU_B * pow(y, (TensorDataT)PU_C) + PU_D);
    return (PU_E * log(y + PU_F) + PU_G);
  }
  
  TensorDataT forward_linear(TensorDataT y)
  {
    return y;
  }
  
  TensorDataT forward_srgb(TensorDataT y)
  {
    if (y <= SRGB_Y0)
      return SRGB_A * y;
    else
      return SRGB_B * pow(y, (TensorDataT)SRGB_C) + SRGB_D;
  }
  
  TensorDataT forward_log(TensorDataT y)
  {
    return log(y + 1.f);
  }
  
  TensorDataT inverse_pu(TensorDataT x)
  {
    if (x <= PU_X0)
      return x / PU_A;
    else if (x <= PU_X1)
      return pow((x - PU_D) / PU_B, 1.f/PU_C);
    else
      return exp((x - PU_G) / PU_E) - PU_F;
  }

  TensorDataT inverse_linear(TensorDataT x)
  {
    return x;
  }

  TensorDataT inverse_log(TensorDataT x)
  {
    return (exp(x) - 1.f);
  }

  TensorDataT inverse_srgb(TensorDataT x)
  {
    if (x <= SRGB_X0)
      return x / SRGB_A;
    else
      return pow((x - SRGB_D) / SRGB_B, 1.f/SRGB_C);
  }
  
private:
  TransferFunctionType type;
  float normScale;
  float rcpNormScale;
};
