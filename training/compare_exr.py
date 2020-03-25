#!/usr/bin/env python

## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os

from config import *
from util import *
from dataset import *
from image import *
from color import *
from ssim import ssim

# Parse the command line arguments
cfg = parse_args(description='Compares two EXR images using the specified quality metrics.')

# Load the images
image1 = load_image(cfg.input[0], num_channels=3)
image2 = load_image(cfg.input[1], num_channels=3)

is_hdr = is_hdr_image(cfg.input[0]) and is_hdr_image(cfg.input[1])

# Load metadata for the images if it exists
tonemap_exposure = cfg.exposure
if os.path.dirname(cfg.input[0]) == os.path.dirname(cfg.input[1]):
  metadata = load_image_metadata(os.path.commonprefix(cfg.input))
  if metadata:
    tonemap_exposure = metadata['exposure']

# Convert the images to tensors
image1 = to_tensor(image1).unsqueeze(0)
image2 = to_tensor(image2).unsqueeze(0)

# Convert the images to sRGB
def to_srgb(image, exposure):
  return srgb_forward(tonemap(image * exposure) if is_hdr else image)

image1 = to_srgb(image1, tonemap_exposure)
image2 = to_srgb(image2, tonemap_exposure)

# Compute the metrics
metric_str = ''
for metric in cfg.metric:
  if metric == 'mse':
    value = ((image1 - image2) ** 2).mean()
  elif metric == 'ssim':
    value = ssim(image1, image2, data_range=1.)
  if metric_str:
    metric_str += ', '
  metric_str += '%s = %.4f' % (metric, value)
if metric_str:
  print(metric_str)
