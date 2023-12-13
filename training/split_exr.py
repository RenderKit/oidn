#!/usr/bin/env python3

## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict
import argparse
import OpenImageIO as oiio

from config import *

def main():
  # Parse the command-line arguments
  cfg = parse_args(description='Splits a multi-channel EXR image into multiple feature images.')

  # Load the input image
  name, ext = os.path.splitext(cfg.input)
  if ext == '.cxr': # Corona EXR
    ext = '.exr'
  if ext.lower() != '.exr':
    error('image must be EXR')

  # Iterate over subimages
  subimage = 0
  while True:
    image = oiio.ImageBuf(cfg.input, subimage, 0)
    if image.has_error:
      error('could not load image')
    subimage += 1

    # Get the channels and group them by layer
    channels = image.spec().channelnames
    if not channels:
      break
    print('Channels:', channels)
    layer_channels = defaultdict(set)
    for channel in channels:
      if len(channel.split('.')) >= 3:
        layer, ch = channel.split('.', 1)
        layer_channels[layer].add(ch)
      else:
        layer_channels[None].add(channel)

    # Set default layer
    if not cfg.layer and len(layer_channels) == 1:
      cfg.layer = list(layer_channels.keys())[0]

    # Extract features
    FEATURES = {
      'hdr' : [
                ('R', 'G', 'B'),
                ('Noisy Image.R', 'Noisy Image.G', 'Noisy Image.B'),
                ('Beauty.R', 'Beauty.G', 'Beauty.B'),
                ('Combined.R', 'Combined.G', 'Combined.B'),
                ('Composite.Combined.R', 'Composite.Combined.G', 'Composite.Combined.B')
              ],
      'a' : [('A',)],
      'alb' : [
                ('albedo.R', 'albedo.G', 'albedo.B'),
                ('Denoising Albedo.R', 'Denoising Albedo.G', 'Denoising Albedo.B'),
                ('ViewLayer.Denoising Albedo.R', 'ViewLayer.Denoising Albedo.G', 'ViewLayer.Denoising Albedo.B'),
                ('VisibleDiffuse.R', 'VisibleDiffuse.G', 'VisibleDiffuse.B'),
                ('diffuse.R', 'diffuse.G', 'diffuse.B'),
                ('DiffCol.R', 'DiffCol.G', 'DiffCol.B'),
                ('albedo.red', 'albedo.green', 'albedo.blue'),
              ],
      'nrm' : [
                ('normal.R', 'normal.G', 'normal.B'),
                ('normal.X', 'normal.Y', 'normal.Z'),
                ('N.R', 'N.G', 'N.B'),
                ('Denoising Normal.X', 'Denoising Normal.Y', 'Denoising Normal.Z'),
                ('ViewLayer.Denoising Normal.X', 'ViewLayer.Denoising Normal.Y', 'ViewLayer.Denoising Normal.Z'),
                ('Normals.R', 'Normals.G', 'Normals.B'),
                ('VisibleNormals.R', 'VisibleNormals.G', 'VisibleNormals.B'),
                ('OptixNormals.R', 'OptixNormals.G', 'OptixNormals.B'),
                ('normal.red', 'normal.green', 'normal.blue'),
              ],
      'z' : [
              ('Denoising Depth.Z',),
              ('ViewLayer.Denoising Depth.Z',)
            ]
    }

    for feature, feature_channel_lists in FEATURES.items():
      for feature_channels in feature_channel_lists:
        # Check whether the feature is present in the selected layer of the image
        if cfg.layer:
          feature_channels = tuple([cfg.layer + '.' + f for f in feature_channels])
        if set(feature_channels).issubset(channels):
          # Save the feature image
          feature_filename = name + '.' + feature + ext
          print(feature_filename)
          new_channels = ('R', 'G', 'B') if len(feature_channels) == 3 else ('Y',)
          feature_image = oiio.ImageBufAlgo.channels(image, feature_channels, new_channels)
          feature_image.spec().attribute('compression', 'piz')
          feature_image.write(feature_filename)
          break

if __name__ == '__main__':
  main()