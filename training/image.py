## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import torch
import OpenImageIO as oiio

from ssim import ssim, ms_ssim
from util import *

## -----------------------------------------------------------------------------
## Image operations
## -----------------------------------------------------------------------------

# Converts a NumPy image to a tensor
def image_to_tensor(image, batch=False):
  # Reorder from HWC to CHW
  tensor = torch.from_numpy(image.transpose((2, 0, 1)))
  if batch:
    return tensor.unsqueeze(0) # reshape to NCHW
  else:
    return tensor

# Converts a tensor to a NumPy image
def tensor_to_image(image):
  if len(image.shape) == 4:
    # Remove N dimension
    image = image.squeeze(0)
  # Reorder from CHW to HWC
  return image.cpu().numpy().transpose((1, 2, 0))

# Computes gradient for a tensor
def tensor_gradient(input):
  input0 = input[..., :-1, :-1]
  didy   = input[..., 1:,  :-1] - input0
  didx   = input[..., :-1, 1:]  - input0
  return torch.cat((didy, didx), -3)

# Compares two image tensors using the specified error metric
def compare_images(a, b, metric='psnr'):
  if metric == 'mse':
    return ((a - b) ** 2).mean()
  elif metric == 'psnr':
    mse = ((a - b) ** 2).mean()
    return 10 * np.log10(1. / mse.item())
  elif metric == 'ssim':
    return ssim(a, b, data_range=1.)
  elif metric == 'msssim':
    return ms_ssim(a, b, data_range=1.)
  else:
    raise ValueError('invalid error metric')

## -----------------------------------------------------------------------------
## Image I/O
## -----------------------------------------------------------------------------

# Loads an image and returns the pixels as a float NumPy array and the number of loaded channels
def load_image(filename, num_channels=None):
  input = oiio.ImageInput.open(filename)
  if not input:
    raise RuntimeError('could not open image: "' + filename + '"')
  load_num_channels = min(input.spec().nchannels, 3)
  if num_channels:
    load_num_channels = min(load_num_channels, num_channels)
  image = input.read_image(subimage=0, miplevel=0, chbegin=0, chend=load_num_channels, format=oiio.FLOAT)
  if image is None:
    raise RuntimeError('could not read image')
  input.close()

  if num_channels and image.shape[2] < num_channels:
    # Repeat the last channel to fill in the missing channels
    repeats = [1] * (image.shape[2] - 1) + [num_channels - image.shape[2] + 1]
    image = np.repeat(image, repeats, axis=2)

  image = np.nan_to_num(image)
  return image, load_num_channels

# Saves a float NumPy image
def save_image(filename, image, num_channels=None):
  if num_channels and num_channels != image.shape[2]:
    if image.shape[2] < num_channels:
      raise RuntimeError('image to save has fewer channels than expected')
    elif num_channels == 1:
      # Compute the average of all channels
      image = np.mean(image, axis=2, keepdims=True)
    else:
      # Truncate to the specified number of channels
      image = image[:, :, :num_channels]

  ext = get_path_ext(filename).lower()
  if ext == 'pfm':
    save_pfm(filename, image)
  elif ext == 'phm':
    save_phm(filename, image)
  else:
    output = oiio.ImageOutput.create(filename)
    if not output:
      raise RuntimeError('could not create image: "' + filename + '"')
    format = oiio.FLOAT if ext == 'exr' else oiio.UINT8
    spec = oiio.ImageSpec(image.shape[1], image.shape[0], image.shape[2], format)
    if ext == 'exr':
      spec.attribute('compression', 'piz')
    elif ext == 'png':
      spec.attribute('png:compressionLevel', 3)
    if not output.open(filename, spec):
      raise RuntimeError('could not open image: "' + filename + '"')
    # FIXME: copy is needed for arrays owned by PyTorch for some reason
    if not output.write_image(image.copy()):
      raise RuntimeError('could not save image')
    output.close()

# Saves a float NumPy image in PFM format
def save_pfm(filename, image):
  with open(filename, 'w') as f:
    num_channels = image.shape[-1]
    if num_channels >= 3:
      f.write('PF\n')
      data = image[..., 0:3]
    else:
      f.write('Pf\n')
      data = image[..., 0]
    data = np.flip(data, 0).astype(np.float32)

    f.write('%d %d\n' % (image.shape[1], image.shape[0]))
    f.write('-1.0\n')
    data.tofile(f)

# Saves a float NumPy image in PHM format
def save_phm(filename, image):
  with open(filename, 'w') as f:
    num_channels = image.shape[-1]
    if num_channels >= 3:
      f.write('PH\n')
      data = image[..., 0:3]
    else:
      f.write('Ph\n')
      data = image[..., 0]
    data = np.flip(data, 0).astype(np.float16)

    f.write('%d %d\n' % (image.shape[1], image.shape[0]))
    f.write('-1.0\n')
    data.tofile(f)