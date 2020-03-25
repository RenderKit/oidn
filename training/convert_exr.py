#!/usr/bin/env python

## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os

from config import *
from util import *
from dataset import *
from image import *
from color import *

# Parse the command line arguments
cfg = parse_args(description='Converts an EXR image to another format, performing tonemapping too if necessary.')

# Load the input image
image = load_image(cfg.input, num_channels=3)

# Load metadata for the image if it exists
tonemap_exposure = cfg.exposure
metadata = load_image_metadata(cfg.input)
if metadata:
  tonemap_exposure = metadata['exposure']

# Convert the image to tensor
image = to_tensor(image).unsqueeze(0)

# Apply tonemapping to the image if needed
if is_hdr_image(cfg.input) and not is_hdr_image(cfg.output):
  image = tonemap(image * tonemap_exposure)

# Convert the image to sRGB if needed
if not is_linear_image(cfg.output):
  image = srgb_forward(image)

# Save the image
save_image(cfg.output, to_numpy(image))