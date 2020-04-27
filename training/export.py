#!/usr/bin/env python3

## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import torch

from config import *
from util import *
from result import *
import tza

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Exports a training result to the runtime model weights format (TZA).')

  # Initialize the PyTorch device
  device = init_device(cfg)

  # Load the checkpoint
  checkpoint = load_checkpoint(cfg, device, cfg.checkpoint)
  epoch = checkpoint['epoch']
  model_state = checkpoint['model_state']

  # Save the weights to a TZA file
  output_filename = os.path.join(get_result_dir(cfg), cfg.result)
  if cfg.checkpoint:
    output_filename += '_%d' % epoch
  output_filename += '.tza'
  print('Saving weights:', output_filename)

  with tza.Writer(output_filename) as output_file:
    for name, value in model_state.items():
      tensor = value.cpu().numpy()
      print(name, tensor.shape)

      if name.endswith('.weight') and len(value.shape) == 4:
        layout = 'oihw'
      elif len(value.shape) == 1:
        layout = 'x'
      else:
        error('unknown state value')

      output_file.write(name, tensor, layout)

if __name__ == '__main__':
  main()