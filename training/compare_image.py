#!/usr/bin/env python3

## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os

from config import *
from util import *
from dataset import *
from image import *
from color import *

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Compares two feature images using the specified quality metrics.')

  # Load the images
  image1 = load_image(cfg.input[0], num_channels=3)
  image2 = load_image(cfg.input[1], num_channels=3)

  feature1 = get_image_feature(cfg.input[0])
  feature2 = get_image_feature(cfg.input[1])
  if feature1 != feature2:
    error('cannot compare different features')

  # Load metadata for the images if it exists
  tonemap_exposure = cfg.exposure
  if os.path.dirname(cfg.input[0]) == os.path.dirname(cfg.input[1]):
    metadata = load_image_metadata(os.path.commonprefix(cfg.input))
    if metadata:
      tonemap_exposure = metadata['exposure']

  # Convert the images to tensors
  image1 = image_to_tensor(image1, batch=True)
  image2 = image_to_tensor(image2, batch=True)

  # Transform the images to sRGB
  image1 = transform_feature(image1, feature1, 'srgb', tonemap_exposure)
  image2 = transform_feature(image2, feature2, 'srgb', tonemap_exposure)

  # Compute the metrics
  metric_str = ''
  for metric in cfg.metric:
    value = compare_images(image1, image2, metric)
    if metric_str:
      metric_str += ', '
    metric_str += '%s = %.4f' % (metric, value)
  if metric_str:
    print(metric_str)

if __name__ == '__main__':
  main()