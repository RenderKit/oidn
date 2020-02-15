// ======================================================================== //
// Copyright 2009-2020 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "image.ih"
#include "color.ih"

static const uniform float HDR_Y_MAX = 65504.f; // maximum HDR value

// -----------------------------------------------------------------------------
// Transfer function
// -----------------------------------------------------------------------------

static void TransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  self->xScale = 1.f;
  self->xRcpScale = 1.f;
  self->exposure = 1.f;
  self->rcpExposure = 1.f;
}

// Computes the normalization scale
static void TransferFunction_initNormalization(uniform TransferFunction* uniform self, uniform float yMax)
{
  const uniform float xMax = extract(reduce_max(self->forward(self, make_vec3f(yMax))), 0);

  self->xScale = 1./xMax;
  self->xRcpScale = xMax;
}

export void TransferFunction_setExposure(uniform TransferFunction* uniform self, uniform float exposure)
{
  self->exposure = exposure;
  self->rcpExposure = (exposure != 0.f) ? (1.f / exposure) : 0.f;
}

// -----------------------------------------------------------------------------
// Transfer function: Linear
// -----------------------------------------------------------------------------

static vec3f LinearTransferFunction_forward(const uniform TransferFunction* uniform self, vec3f y)
{
  return min(y, 1.f);
}

static vec3f LinearTransferFunction_inverse(const uniform TransferFunction* uniform self, vec3f x)
{
  return min(x, 1.f);
}

export void LinearTransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  TransferFunction_Constructor(self);
  self->forward = LinearTransferFunction_forward;
  self->inverse = LinearTransferFunction_inverse;
}

// -----------------------------------------------------------------------------
// Transfer function: Gamma
// -----------------------------------------------------------------------------

static vec3f GammaTransferFunction_forward(const uniform TransferFunction* uniform self, vec3f y)
{
  return min(pow(y, 1.f/2.2f), 1.f);
}

static vec3f GammaTransferFunction_inverse(const uniform TransferFunction* uniform self, vec3f x)
{
  return min(pow(x, 2.2f), 1.f);
}

export void GammaTransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  TransferFunction_Constructor(self);
  self->forward = GammaTransferFunction_forward;
  self->inverse = GammaTransferFunction_inverse;
}

// -----------------------------------------------------------------------------
// Transfer function: Log
// -----------------------------------------------------------------------------

static vec3f LogTransferFunction_forward(const uniform TransferFunction* uniform self, vec3f y)
{
  return log(y * self->exposure + 1.f) * self->xScale;
}

static vec3f LogTransferFunction_inverse(const uniform TransferFunction* uniform self, vec3f x)
{
  return (exp(x * self->xRcpScale) - 1.f) * self->rcpExposure;
}

export void LogTransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  TransferFunction_Constructor(self);
  self->forward = LogTransferFunction_forward;
  self->inverse = LogTransferFunction_inverse;
  TransferFunction_initNormalization(self, HDR_Y_MAX);
}

// -----------------------------------------------------------------------------
// Transfer function: PU2
// -----------------------------------------------------------------------------

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
  y = y * self->exposure;
  return make_vec3f(puForward(y.x), puForward(y.y), puForward(y.z)) * self->xScale;
}

static vec3f PUTransferFunction_inverse(const uniform TransferFunction* uniform self, vec3f x)
{
  x = x * self->xRcpScale;
  return make_vec3f(puInverse(x.x), puInverse(x.y), puInverse(x.z)) * self->rcpExposure;
}

export void PUTransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  TransferFunction_Constructor(self);
  self->forward = PUTransferFunction_forward;
  self->inverse = PUTransferFunction_inverse;
  TransferFunction_initNormalization(self, HDR_Y_MAX);
}

// -----------------------------------------------------------------------------
// Autoexposure
// -----------------------------------------------------------------------------

// Returns the average luminance of the specified image tile
export uniform float getAvgLuminance(const uniform Image& color,
                                     uniform int beginH, uniform int endH,
                                     uniform int beginW, uniform int endW)
{
  float L = 0.f;

  for (uniform int h = beginH; h < endH; ++h)
  {
    foreach (w = beginW ... endW)
    {
      vec3f c = get3f(color, h, w);
      c = select(isfinite(c), max(c, 0.f), make_vec3f(0.f)); // sanitize
      L += luminance(c);
    }
  }

  return reduce_add(L) / ((endH - beginH) * (endW - beginW));
}
