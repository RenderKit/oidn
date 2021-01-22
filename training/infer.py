#!/usr/bin/env python3

## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import time
import numpy as np
import torch

from config import *
from util import *
from dataset import *
from model import *
from color import *
from result import *

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Performs inference on a dataset using the specified training result.')

  # Initialize the PyTorch device
  device = init_device(cfg)

  # Open the result
  result_dir = get_result_dir(cfg)
  if not os.path.isdir(result_dir):
    error('result does not exist')
  print('Result:', cfg.result)

  # Load the result config
  result_cfg = load_config(result_dir)
  cfg.features = result_cfg.features
  cfg.transfer = result_cfg.transfer
  cfg.model    = result_cfg.model
  main_feature = get_main_feature(cfg.features)
  num_main_channels = len(get_model_channels(main_feature))

  # Initialize the dataset
  data_dir = get_data_dir(cfg, cfg.input_data)
  image_sample_groups = get_image_sample_groups(data_dir, cfg.features)

  # Initialize the model
  model = get_model(cfg)
  model.to(device)

  # Load the checkpoint
  checkpoint = load_checkpoint(result_dir, device, cfg.num_epochs, model)
  epoch = checkpoint['epoch']
  print('Epoch:', epoch)

  # Initialize the transfer function
  transfer = get_transfer_function(cfg)

  # Iterate over the images
  print()
  output_dir = os.path.join(cfg.output_dir, cfg.input_data)
  metric_sum = {metric : 0. for metric in cfg.metric}
  metric_count = 0
  model.eval()

  # Inference function
  def infer(input, exposure):
    x = input.clone()

    # Apply the transfer function
    if transfer:
      color = x[:, 0:num_main_channels, ...]
      if main_feature == 'hdr':
        color *= exposure
      color = transfer.forward(color)
      x[:, 0:num_main_channels, ...] = color

    # Pad the output
    shape = x.shape
    x = F.pad(x, (0, round_up(shape[3], model.alignment) - shape[3],
                  0, round_up(shape[2], model.alignment) - shape[2]))

    # Run the inference
    x = model(x)

    # Unpad the output
    x = x[:, :, :shape[2], :shape[3]]

    # Sanitize the output
    x = torch.clamp(x, min=0.)

    # Apply the inverse transfer function
    if transfer:
      x = transfer.inverse(x)
      if main_feature == 'hdr':
        x /= exposure
      else:
        x = torch.clamp(x, max=1.)
        
    return x

  # Saves an image in different formats
  def save_images(path, image, image_srgb):
    image      = tensor_to_image(image)
    image_srgb = tensor_to_image(image_srgb)
    filename_prefix = path + '.' + main_feature + '.'
    for format in cfg.format:
      if format in {'exr', 'pfm', 'hdr'}:
        # Transform to original range
        if main_feature in {'shl1', 'nrm'}:
          image = image * 2. - 1. # [0..1] -> [-1..1]
        save_image(filename_prefix + format, image)
      else:
        save_image(filename_prefix + format, image_srgb)

  with torch.no_grad():
    for group, input_names, target_name in image_sample_groups:
      # Create the output directory if it does not exist
      output_group_dir = os.path.join(output_dir, os.path.dirname(group))
      if not os.path.isdir(output_group_dir):
        os.makedirs(output_group_dir)

      # Load metadata for the images if it exists
      tonemap_exposure = 1.
      metadata = load_image_metadata(os.path.join(data_dir, group))
      if metadata:
        tonemap_exposure = metadata['exposure']
        save_image_metadata(os.path.join(output_dir, group), metadata)

      # Load the target image if it exists
      if target_name:
        target = load_image_features(os.path.join(data_dir, target_name), main_feature)
        target = image_to_tensor(target, batch=True).to(device)
        target_srgb = transform_feature(target, main_feature, 'srgb', tonemap_exposure)

      # Iterate over the input images
      for input_name in input_names:
        print(input_name, '...', end='', flush=True)

        # Load the input image
        input = load_image_features(os.path.join(data_dir, input_name), cfg.features)

        # Compute the autoexposure value
        exposure = autoexposure(input) if main_feature == 'hdr' else 1.

        # Infer
        input = image_to_tensor(input, batch=True).to(device)
        output = infer(input, exposure)

        input = input[:, 0:num_main_channels, ...] # keep only the main feature
        input_srgb  = transform_feature(input,  main_feature, 'srgb', tonemap_exposure)
        output_srgb = transform_feature(output, main_feature, 'srgb', tonemap_exposure)

        # Compute metrics
        metric_str = ''
        if target_name and cfg.metric:
          for metric in cfg.metric:
            value = compare_images(output_srgb, target_srgb, metric)
            metric_sum[metric] += value
            if metric_str:
              metric_str += ', '
            metric_str += f'{metric}={value:.4f}'
          metric_count += 1

        # Save the input and output images
        output_name = input_name + '.' + cfg.result
        if cfg.num_epochs:
          output_name += f'_{epoch}'
        if cfg.save_all:
          save_images(os.path.join(output_dir, input_name), input, input_srgb)
        save_images(os.path.join(output_dir, output_name), output, output_srgb)

        # Print metrics
        if metric_str:
          metric_str = ' ' + metric_str
        print(metric_str)

      # Save the target image if it exists
      if cfg.save_all and target_name:
        save_images(os.path.join(output_dir, target_name), target, target_srgb)

  # Print summary
  if metric_count > 0:
    metric_str = ''
    for metric in cfg.metric:
      value = metric_sum[metric] / metric_count
      if metric_str:
        metric_str += ', '
      metric_str += f'{metric}_avg={value:.4f}'
    print()
    print(f'{cfg.result}: {metric_str} ({metric_count} images)')

if __name__ == '__main__':
  main()