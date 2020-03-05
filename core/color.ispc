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

static void TransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  self->xScale    = 1.f;
  self->xRcpScale = 1.f;
  self->yScale    = 1.f;
  self->yRcpScale = 1.f;
}

// Computes the normalization scale
static void TransferFunction_initNormalization(uniform TransferFunction* uniform self, uniform float yMax)
{
  const uniform float xMax = extract(reduce_max(self->forward(self, make_vec3f(yMax))), 0);

  self->xScale    = 1./xMax;
  self->xRcpScale = xMax;
}

export void TransferFunction_setExposure(uniform TransferFunction* uniform self, uniform float exposure)
{
  self->yScale    = exposure;
  self->yRcpScale = (exposure != 0.f) ? (1.f / exposure) : 0.f;
}

// -----------------------------------------------------------------------------
// Linear transfer function
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
// Gamma transfer function
// -----------------------------------------------------------------------------

static vec3f GammaTransferFunction_forward(const uniform TransferFunction* uniform self, vec3f y)
{
  return pow(y, 1.f/2.2f);
}

static vec3f GammaTransferFunction_inverse(const uniform TransferFunction* uniform self, vec3f x)
{
  return pow(x, 2.2f);
}

export void GammaTransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  TransferFunction_Constructor(self);
  self->forward = GammaTransferFunction_forward;
  self->inverse = GammaTransferFunction_inverse;
}

// -----------------------------------------------------------------------------
// Log transfer function
// -----------------------------------------------------------------------------

static vec3f LogTransferFunction_forward(const uniform TransferFunction* uniform self, vec3f y)
{
  return log(y * self->yScale + 1.f) * self->xScale;
}

static vec3f LogTransferFunction_inverse(const uniform TransferFunction* uniform self, vec3f x)
{
  return (exp(x * self->xRcpScale) - 1.f) * self->yRcpScale;
}

export void LogTransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  TransferFunction_Constructor(self);
  self->forward = LogTransferFunction_forward;
  self->inverse = LogTransferFunction_inverse;
  TransferFunction_initNormalization(self, HDR_Y_MAX);
}

// -----------------------------------------------------------------------------
// PQX transfer function
// -----------------------------------------------------------------------------

static const uniform float PQ_M1 = 2610.f / 4096.f / 4.f;
static const uniform float PQ_M2 = 2523.f / 4096.f * 128.f;
static const uniform float PQ_C1 = 3424.f / 4096.f;
static const uniform float PQ_C2 = 2413.f / 4096.f * 32.f;
static const uniform float PQ_C3 = 2392.f / 4096.f * 32.f;
static const uniform float PQX_K = 3711.f / 4096.f / 8.f;

static const uniform float PQX_Y_SCALE = 100.f / 10000.f;

inline float pqForward(float y)
{
  const float yp = pow(y, PQ_M1);
  return pow((PQ_C1 + PQ_C2 * yp) * rcp(1.f + PQ_C3 * yp), PQ_M2);
}

inline float pqxForward(float y)
{
  if (y <= 1.f)
    return pqForward(y);
  else
    return PQX_K * log(y) + 1.f;
}

inline float pqInverse(float x)
{
  const float xp = pow(x, 1.f/PQ_M2);
  return pow(max((xp - PQ_C1) * rcp(PQ_C2 - PQ_C3 * xp), 0.f), 1.f/PQ_M1);
}

inline float pqxInverse(float x)
{
  if (x <= 1.f)
    return pqInverse(x);
  else
    return exp((x - 1.f) * (1.f/PQX_K));
}

static vec3f PQXTransferFunction_forward(const uniform TransferFunction* uniform self, vec3f y)
{
  y = y * self->yScale * PQX_Y_SCALE;
  return make_vec3f(pqxForward(y.x), pqxForward(y.y), pqxForward(y.z)) * self->xScale;
}

static vec3f PQXTransferFunction_inverse(const uniform TransferFunction* uniform self, vec3f x)
{
  x = x * self->xRcpScale;
  return make_vec3f(pqxInverse(x.x), pqxInverse(x.y), pqxInverse(x.z)) * (1./PQX_Y_SCALE) * self->yRcpScale;
}

export void PQXTransferFunction_Constructor(uniform TransferFunction* uniform self)
{
  TransferFunction_Constructor(self);
  self->forward = PQXTransferFunction_forward;
  self->inverse = PQXTransferFunction_inverse;
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
      c = max(inf_to_zero(c), 0.f); // sanitize
      L += luminance(c);
    }
  }

  return reduce_add(L) / ((endH - beginH) * (endW - beginW));
}
