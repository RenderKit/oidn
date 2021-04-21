#!/usr/bin/env python3

## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os

from config import *
from util import *
from dataset import *
from image import *

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
  image = image_to_tensor(image, batch=True)

  # Transform the image
  input_feature  = get_image_feature(cfg.input)
  output_feature = get_image_feature(cfg.output)
  image = transform_feature(image, input_feature, output_feature, tonemap_exposure)

  # Save the image
  save_image(cfg.output, tensor_to_image(image))

if __name__ == '__main__':
  main()