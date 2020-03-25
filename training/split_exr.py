#!/usr/bin/env python

## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import argparse
import OpenImageIO as oiio

from config import *

# Parse the command-line arguments
cfg = parse_args(description='Splits a multi-channel EXR image into multiple AOV images.')

# Load the input image
in_name, in_ext = os.path.splitext(cfg.input)
in_image = oiio.ImageBuf(cfg.input)
if in_image.has_error:
  error('could not load image')
in_spec = in_image.spec()
in_channels = in_spec.channelnames
#print(in_channels)

# Extract AOVs
AOVS = {
  'hdr' : [
            ('R', 'G', 'B'),
            ('View Layer.Combined.R', 'View Layer.Combined.G', 'View Layer.Combined.B')
          ],
  'alb' : [
            ('albedo.R', 'albedo.G', 'albedo.B'),
            ('View Layer.Denoising Albedo.R', 'View Layer.Denoising Albedo.G', 'View Layer.Denoising Albedo.B')
          ],
  'nrm' : [
            ('normal.R', 'normal.G', 'normal.B'),
            ('N.R', 'N.G', 'N.B'),
            ('View Layer.Denoising Normal.X', 'View Layer.Denoising Normal.Y', 'View Layer.Denoising Normal.Z')
          ]
}

for aov, aov_channel_lists in AOVS.items():
  for aov_channels in aov_channel_lists:
    # Check whether the AOV is present in the input image
    if set(aov_channels).issubset(in_channels):
      # Save the AOV image
      aov_filename = in_name + '.' + aov + in_ext
      print(aov_filename)
      new_channels = ('R', 'G', 'B') if len(aov_channels) == 3 else ('Y')
      aov_image = oiio.ImageBufAlgo.channels(in_image, aov_channels, new_channels)
      aov_image.write(aov_filename)
      break