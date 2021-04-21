#!/usr/bin/env python

## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import argparse
from common import *

MODELS = [
  'rt_hdr_alb_nrm',
  'rt_hdr_alb',
  'rt_hdr',
  'rt_ldr_alb_nrm',
  'rt_ldr_alb',
  'rt_ldr',
  'rtlightmap_hdr'
]

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Builds the weights blobs from the training results.')
parser.usage = '\rIntel(R) Open Image Denoise - Build Weights\n' + parser.format_usage()
parser.add_argument('--results_dir', '-R', type=str, default=os.path.join(root_dir, 'training', 'results'), help='directory of training results')
cfg = parser.parse_args()

weights_dir = os.path.join(root_dir, 'weights')
export_cmd = os.path.join(root_dir, 'training', 'export.py')

# Export the weights blobs
for model in MODELS:
  tza_filename = os.path.join(weights_dir, model + '.tza')
  run(export_cmd + f' -R {cfg.results_dir} -r {model} -o {tza_filename}')
  print()