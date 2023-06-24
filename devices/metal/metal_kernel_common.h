// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

enum TransferFunctionType {
  Linear,
  SRGB,
  PU,
  Log
};

enum KernelDataType {
  f16,
  f32
};

struct Tile
{
  int32_t hSrcBegin;
  int32_t wSrcBegin;
  int32_t hDstBegin;
  int32_t wDstBegin;
  int32_t H;
  int32_t W;
};

struct ProcessParams {
  float                 inputScale;
  float                 outputScale;
    
  float                 normScale;
  
  bool                  snorm;
  bool                  hdr;
  
  TransferFunctionType  func;
  
  int32_t               C;
  int32_t               H;
  int32_t               W;
  
  Tile                  tile;
  
  bool                  color;
  bool                  albedo;
  bool                  normal;
    
  KernelDataType        inputDataType;
  KernelDataType        outputDataType;
};

struct AutoexposureParams {
  int32_t               H;
  int32_t               W;
  
  int32_t               numBinsH;
  int32_t               numBinsW;
  
  int32_t               maxBinSize;
  
  KernelDataType        inputDataType;
};

float nan_to_zero(float value);

template<typename F, typename T>
T castTo(F ptr)
{
  return static_cast<T>(ptr);
}
