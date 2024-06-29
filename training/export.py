#!/usr/bin/env python3

## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
from glob import glob
import numpy as np
import torch

from config import *
from util import *
from model import *
from result import *
import tza

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Exports a trained model to the runtime weights (TZA) or some other format.')

  print('Result:', cfg.result)

  if cfg.target == 'package':
    # Get the output filename
    if cfg.output:
      output_filename = cfg.output
    else:
      output_filename = os.path.join(cfg.results_dir, cfg.result) + '.zip'
    print('Output:', output_filename)

    # Get the list of files that belong to the result (latest checkpoint only)
    result_dir = get_result_dir(cfg)
    filenames = [get_config_filename(result_dir)]
    filenames.append(get_checkpoint_state_filename(result_dir))
    latest_epoch = get_latest_checkpoint_epoch(result_dir)
    filenames.append(get_checkpoint_filename(result_dir, latest_epoch))
    filenames += glob(os.path.join(get_result_log_dir(result_dir), 'events.out.*'))
    filenames += glob(os.path.join(result_dir, 'src.*'))

    # Save the ZIP file
    save_zip(output_filename, filenames, root_dir=cfg.results_dir)
  else:
    # Initialize the PyTorch device
    device = init_device(cfg)

    # Load the result config
    result_dir = get_result_dir(cfg)
    if not os.path.isdir(result_dir):
      error('result does not exist')
    result_cfg = load_config(result_dir)

    # Initialize the model
    if cfg.target in {'onnx', 'onnx_noparams'}:
      model = get_model(result_cfg)
      model.to(device)
    else:
      model = None

    # Load the checkpoint
    checkpoint = load_checkpoint(result_dir, device, cfg.num_epochs, model)
    epoch = checkpoint['epoch']
    model_state = checkpoint['model_state']
    print('Epoch:', epoch)

    if cfg.target == 'weights':
      # Save the weights to a TZA file
      if cfg.output:
        output_filename = cfg.output
      else:
        output_filename = os.path.join(result_dir, cfg.result)
        if cfg.num_epochs:
          output_filename += '_%d' % epoch
        output_filename += '.tza'
      print('Output:', output_filename)
      print()

      with tza.Writer(output_filename) as output_file:
        for name, value in model_state.items():
          tensor = value.half()
          tensor = tensor.cpu().numpy()
          print(name, tensor.shape)

          if name.endswith('.weight') and len(value.shape) == 4:
            layout = 'oihw'
          elif len(value.shape) == 1:
            layout = 'x'
          else:
            error('unknown state value')

          output_file.write(name, tensor, layout)
    elif cfg.target in {'onnx', 'onnx_noparams'}:
      # Export the model to ONNX
      if cfg.output:
        output_filename = cfg.output
      else:
        output_filename = os.path.join(result_dir, cfg.result)
        if cfg.target != 'onnx_noparams' and cfg.num_epochs:
          output_filename += '_%d' % epoch
        output_filename += '.onnx'
      print('Output:', output_filename)
      print()

      W, H = 1920, 1080
      C = len(get_model_channels(result_cfg.features))
      dtype = torch.float32 if device.type == 'cpu' else torch.float16
      input_shape = [1, C, round_up(H, model.alignment), round_up(W, model.alignment)]
      input = torch.zeros(input_shape, dtype=dtype, device=device)
      model.to(dtype=dtype)

      torch.onnx.export(model, input, output_filename,
                        opset_version=11,
                        export_params=(cfg.target != 'onnx_noparams'),
                        input_names=['input'],
                        output_names=['output'])

if __name__ == '__main__':
  main()