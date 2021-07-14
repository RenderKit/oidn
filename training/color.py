## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from util import *

HDR_Y_MAX = 65504. # maximum HDR value

# Computes the luminance of an RGB color
def luminance(r, g, b):
  return 0.212671 * r + 0.715160 * g + 0.072169 * b

## -----------------------------------------------------------------------------
## Transfer function
## -----------------------------------------------------------------------------

class TransferFunction: pass

def get_transfer_function(cfg):
  type = cfg.transfer
  if type == 'linear':
    return LinearTransferFunction()
  elif type == 'srgb':
    return SRGBTransferFunction()
  elif type == 'pu':
    return PUTransferFunction()
  elif type == 'log':
    return LogTransferFunction()
  else:
    error('invalid transfer function')

## -----------------------------------------------------------------------------
## Transfer function: Linear
## -----------------------------------------------------------------------------

class LinearTransferFunction(TransferFunction):
  def forward(self, y):
    return y

  def inverse(self, x):
    return x

## -----------------------------------------------------------------------------
## Transfer function: sRGB
## -----------------------------------------------------------------------------

SRGB_A  =  12.92
SRGB_B  =  1.055
SRGB_C  =  1./2.4
SRGB_D  = -0.055
SRGB_Y0 =  0.0031308
SRGB_X0 =  0.04045

def srgb_forward(y):
  return torch.where(y <= SRGB_Y0,
                     SRGB_A * y,
                     SRGB_B * torch.pow(y, SRGB_C) + SRGB_D)

def srgb_inverse(x):
  return torch.where(x <= SRGB_X0,
                     x / SRGB_A,
                     torch.pow((x - SRGB_D) / SRGB_B, 1./SRGB_C))

class SRGBTransferFunction(TransferFunction):
  def forward(self, y):
    return srgb_forward(y)

  def inverse(self, x):
    return srgb_inverse(x)

## -----------------------------------------------------------------------------
## Transfer function: PU
## -----------------------------------------------------------------------------

# Fit of PU2 curve normalized at 100 cd/m^2
# [Aydin et al., 2008, "Extending Quality Metrics to Full Luminance Range Images"]
PU_A  =  1.41283765e+03
PU_B  =  1.64593172e+00
PU_C  =  4.31384981e-01
PU_D  = -2.94139609e-03
PU_E  =  1.92653254e-01
PU_F  =  6.26026094e-03
PU_G  =  9.98620152e-01
PU_Y0 =  1.57945760e-06
PU_Y1 =  3.22087631e-02
PU_X0 =  2.23151711e-03
PU_X1 =  3.70974749e-01

def pu_forward(y):
  return torch.where(y <= PU_Y0,
                     PU_A * y,
                     torch.where(y <= PU_Y1,
                                 PU_B * torch.pow(y, PU_C)  + PU_D,
                                 PU_E * torch.log(y + PU_F) + PU_G))

def pu_inverse(x):
  return torch.where(x <= PU_X0,
                     x / PU_A,
                     torch.where(x <= PU_X1,
                                 torch.pow((x - PU_D) / PU_B, 1./PU_C),
                                 torch.exp((x - PU_G) / PU_E) - PU_F))

PU_NORM_SCALE = 1. / pu_forward(torch.tensor(HDR_Y_MAX)).item()

class PUTransferFunction(TransferFunction):
  def forward(self, y):
    return pu_forward(y) * PU_NORM_SCALE

  def inverse(self, x):
    return pu_inverse(x / PU_NORM_SCALE)

## -----------------------------------------------------------------------------
## Transfer function: Log
## -----------------------------------------------------------------------------

def log_forward(y):
  return torch.log(y + 1.)

def log_inverse(x):
  return torch.exp(x) - 1.

LOG_NORM_SCALE = 1. / log_forward(torch.tensor(HDR_Y_MAX)).item()

class LogTransferFunction(TransferFunction):
  def forward(self, y):
    return log_forward(y) * LOG_NORM_SCALE

  def inverse(self, x):
    return log_inverse(x / LOG_NORM_SCALE)

## -----------------------------------------------------------------------------
## Autoexposure
## -----------------------------------------------------------------------------

# Computes an autoexposure value for a NumPy image
def autoexposure(image):
  key = 0.18
  eps = 1e-8
  K = 16 # downsampling amount

  # Compute the luminance of each pixel
  r = image[..., 0]
  g = image[..., 1]
  b = image[..., 2]
  L = luminance(r, g, b)

  # Downsample the image to minimize sensitivity to noise
  H = L.shape[0] # original height
  W = L.shape[1] # original width
  HK = (H + K//2) // K # downsampled height
  WK = (W + K//2) // K # downsampled width

  LK = np.zeros((HK, WK), dtype=L.dtype)
  for i in range(HK):
    for j in range(WK):
      beginH = i     * H // HK
      beginW = j     * W // WK
      endH   = (i+1) * H // HK
      endW   = (j+1) * W // WK

      LK[i, j] = L[beginH:endH, beginW:endW].mean()

  L = LK

  # Keep only values greater than epsilon
  L = L[L > eps]
  if L.size == 0:
    return 1.

  # Compute the exposure value
  return float(key / np.exp2(np.log2(L).mean()))

## -----------------------------------------------------------------------------
## Tonemapping
## -----------------------------------------------------------------------------

# Filmic tonemapping operator
# [Hable, 2010, "Uncharted 2: HDR Lighting"]
def tonemap(x):
  A = 0.22
  B = 0.30
  C = 0.10
  D = 0.20
  E = 0.01
  F = 0.30
  W = 11.2
  scale = 1.758141 # exposure bias to match 18% middle gray

  def eval(x):
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F

  return torch.clamp(eval(x * scale) / eval(W), max=1.)