// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image.isph"
#include "color.isph"

static const uniform float HDR_Y_MAX = 65504.f; // maximum HDR value

// -----------------------------------------------------------------------------
// Transfer function
// -----------------------------------------------------------------------------

static void TransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  self->inputScale   = 1.f;
  self->outputScale  = 1.f;

  self->normScale    = 1.f;
  self->rcpNormScale = 1.f;
}

// Computes the normalization scale
static void TransferFunction_initNormalization(uniform TransferFunction* uniform self, uniform float yMax)
{
  const uniform float xMax = extract(reduce_max(self->forward(self, make_vec3f(yMax))), 0);

  self->normScale    = 1./xMax;
  self->rcpNormScale = xMax;
}

// -----------------------------------------------------------------------------
// Transfer function: Linear
// -----------------------------------------------------------------------------

static vec3f LinearTransferFunction_forward(const uniform TransferFunction* uniform self, vec3f y)
{
  return y;
}

static vec3f LinearTransferFunction_inverse(const uniform TransferFunction* uniform self, vec3f x)
{
  return x;
}

export void LinearTransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  TransferFunction_Constructor(self);
  self->forward = LinearTransferFunction_forward;
  self->inverse = LinearTransferFunction_inverse;
}

// -----------------------------------------------------------------------------
// Transfer function: sRGB
// -----------------------------------------------------------------------------

static const uniform float SRGB_A  =  12.92f;
static const uniform float SRGB_B  =  1.055f;
static const uniform float SRGB_C  =  1.f/2.4f;
static const uniform float SRGB_D  = -0.055f;
static const uniform float SRGB_Y0 =  0.0031308f;
static const uniform float SRGB_X0 =  0.04045f;

inline float srgbForward(float y)
{
  if (y <= SRGB_Y0)
    return SRGB_A * y;
  else
    return SRGB_B * pow(y, SRGB_C) + SRGB_D;
}

inline float srgbInverse(float x)
{
  if (x <= SRGB_X0)
    return x / SRGB_A;
  else
    return pow((x - SRGB_D) / SRGB_B, 1.f/SRGB_C);
}

static vec3f SRGBTransferFunction_forward(const uniform TransferFunction* uniform self, vec3f y)
{
  return make_vec3f(srgbForward(y.x), srgbForward(y.y), srgbForward(y.z));
}

static vec3f SRGBTransferFunction_inverse(const uniform TransferFunction* uniform self, vec3f x)
{
  return make_vec3f(srgbInverse(x.x), srgbInverse(x.y), srgbInverse(x.z));
}

export void SRGBTransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  TransferFunction_Constructor(self);
  self->forward = SRGBTransferFunction_forward;
  self->inverse = SRGBTransferFunction_inverse;
}

// -----------------------------------------------------------------------------
// Transfer function: PU
// -----------------------------------------------------------------------------

// Fit of PU2 curve normalized at 100 cd/m^2
// [Aydin et al., 2008, "Extending Quality Metrics to Full Luminance Range Images"]
static const uniform float PU_A  =  1.41283765e+03f;
static const uniform float PU_B  =  1.64593172e+00f;
static const uniform float PU_C  =  4.31384981e-01f;
static const uniform float PU_D  = -2.94139609e-03f;
static const uniform float PU_E  =  1.92653254e-01f;
static const uniform float PU_F  =  6.26026094e-03f;
static const uniform float PU_G  =  9.98620152e-01f;
static const uniform float PU_Y0 =  1.57945760e-06f;
static const uniform float PU_Y1 =  3.22087631e-02f;
static const uniform float PU_X0 =  2.23151711e-03f;
static const uniform float PU_X1 =  3.70974749e-01f;

inline float puForward(float y)
{
  if (y <= PU_Y0)
    return PU_A * y;
  else if (y <= PU_Y1)
    return PU_B * pow(y, PU_C) + PU_D;
  else
    return PU_E * log(y + PU_F) + PU_G;
}

inline float puInverse(float x)
{
  if (x <= PU_X0)
    return x / PU_A;
  else if (x <= PU_X1)
    return pow((x - PU_D) / PU_B, 1.f/PU_C);
  else
    return exp((x - PU_G) / PU_E) - PU_F;
}

static vec3f PUTransferFunction_forward(const uniform TransferFunction* uniform self, vec3f y)
{
  return make_vec3f(puForward(y.x), puForward(y.y), puForward(y.z)) * self->normScale;
}

static vec3f PUTransferFunction_inverse(const uniform TransferFunction* uniform self, vec3f x)
{
  x = x * self->rcpNormScale;
  return make_vec3f(puInverse(x.x), puInverse(x.y), puInverse(x.z));
}

export void PUTransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  TransferFunction_Constructor(self);
  self->forward = PUTransferFunction_forward;
  self->inverse = PUTransferFunction_inverse;
  TransferFunction_initNormalization(self, HDR_Y_MAX);
}

// -----------------------------------------------------------------------------
// Transfer function: Log
// -----------------------------------------------------------------------------

static vec3f LogTransferFunction_forward(const uniform TransferFunction* uniform self, vec3f y)
{
  return log(y + 1.f) * self->normScale;
}

static vec3f LogTransferFunction_inverse(const uniform TransferFunction* uniform self, vec3f x)
{
  return (exp(x * self->rcpNormScale) - 1.f);
}

export void LogTransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  TransferFunction_Constructor(self);
  self->forward = LogTransferFunction_forward;
  self->inverse = LogTransferFunction_inverse;
  TransferFunction_initNormalization(self, HDR_Y_MAX);
}

// -----------------------------------------------------------------------------
// Autoexposure
// -----------------------------------------------------------------------------

// Returns the average luminance of the specified image tile
export uniform float getAvgLuminance(const uniform ImageAccessor& color,
                                     uniform int beginH, uniform int endH,
                                     uniform int beginW, uniform int endW)
{
  float L = 0.f;

  for (uniform int h = beginH; h < endH; ++h)
  {
    foreach (w = beginW ... endW)
    {
      vec3f c = get3f(color, h, w);
      c = clamp(nan_to_zero(c), 0.f, pos_max); // sanitize
      L += luminance(c);
    }
  }

  return reduce_add(L) / ((endH - beginH) * (endW - beginW));
}
