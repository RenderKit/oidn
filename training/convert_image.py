#!/usr/bin/env python3

## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os

from config import *
from util import *
from dataset import *
from image import *
from color import *

# Transforms a feature image to another feature type
def transform_image(image, input_feature, output_feature, exposure=1.):
  if input_feature == 'hdr' and output_feature in {'ldr', None}:
    image = tonemap(image * exposure)
  if not output_feature:
    # Transform to sRGB
    if input_feature in {'hdr', 'ldr', 'alb'}:
      image = srgb_forward(image)
    elif input_feature == 'nrm':
      # Transform [-1, 1] -> [0, 1]
      image = image * 0.5 + 0.5
  return image

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Converts a feature image to a different image format.')

  # Load the input image
  image = load_image(cfg.input, num_channels=3)

  # Load metadata for the image if it exists
  tonemap_exposure = cfg.exposure
  metadata = load_image_metadata(cfg.input)
  if metadata:
    tonemap_exposure = metadata['exposure']

  # Convert the image to tensor
  image = to_tensor(image).unsqueeze(0)

  # Transform the image
  input_feature  = get_image_feature(cfg.input)
  output_feature = get_image_feature(cfg.output)
  image = transform_image(image, input_feature, output_feature, tonemap_exposure)

  # Save the image
  save_image(cfg.output, to_numpy(image))

if __name__ == '__main__':
  main()