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

# Inference function object
class Infer(object):
  def __init__(self, cfg, device, result=None):
    # Load the result config
    result_dir = get_result_dir(cfg, result)
    if not os.path.isdir(result_dir):
      error('result does not exist')
    result_cfg = load_config(result_dir)
    self.features = result_cfg.features
    self.main_feature = get_main_feature(self.features)
    self.aux_features = get_aux_features(self.features)
    self.all_channels = get_dataset_channels(self.features)
    self.num_main_channels = len(get_dataset_channels(self.main_feature))

    # Initialize the model
    self.model = get_model(result_cfg)
    self.model.to(device)

    # Load the checkpoint
    checkpoint = load_checkpoint(result_dir, device, cfg.num_epochs, self.model)
    self.epoch = checkpoint['epoch']

    # Initialize the transfer function
    self.transfer = get_transfer_function(result_cfg)

    # Set the model to evaluation mode
    self.model.eval()

    # Initialize auxiliary feature inference
    self.aux_infers = {}
    if self.aux_features:
      for aux_result in set(cfg.aux_results):
        aux_infer = Infer(cfg, device, aux_result)
        if (aux_infer.main_feature not in self.aux_features) or aux_infer.aux_features:
          error(f'result {aux_result} does not correspond to an auxiliary feature')
        self.aux_infers[aux_infer.main_feature] = aux_infer

  # Inference function
  def __call__(self, input, exposure=1.):
    image = input.clone()

    # Apply the transfer function
    color = image[:, 0:self.num_main_channels, ...]
    if self.main_feature == 'hdr':
      color *= exposure
    color = self.transfer.forward(color)
    image[:, 0:self.num_main_channels, ...] = color

    # Pad the output
    shape = image.shape
    image = F.pad(image, (0, round_up(shape[3], self.model.alignment) - shape[3],
                          0, round_up(shape[2], self.model.alignment) - shape[2]))

    # Prefilter the auxiliary features
    for aux_feature, aux_infer in self.aux_infers.items():
      aux_channels = get_dataset_channels(aux_feature)
      aux_channel_indices = get_channel_indices(aux_channels, self.all_channels)
      aux = image[:, aux_channel_indices, ...]
      aux = aux_infer(aux)
      image[:, aux_channel_indices, ...] = aux

    # Filter the main feature
    if self.main_feature == 'sh1':
      # Iterate over x, y, z
      image = torch.cat([self.model(torch.cat((image[:, i:i+3, ...], image[:, 9:, ...]), 1)) for i in [0, 3, 6]], 1)
    else:
      image = self.model(image)

    # Unpad the output
    image = image[:, :, :shape[2], :shape[3]]

    # Sanitize the output
    image = torch.clamp(image, min=0.)

    # Apply the inverse transfer function
    image = self.transfer.inverse(image)
    if self.main_feature == 'hdr':
      image /= exposure
    else:
      image = torch.clamp(image, max=1.)
        
    return image

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Performs inference on a dataset using the specified training result.')

  # Initialize the PyTorch device
  device = init_device(cfg)

  # Initialize the inference function
  infer = Infer(cfg, device)
  print('Result:', cfg.result)
  print('Epoch:', infer.epoch)

  # Initialize the dataset
  data_dir = get_data_dir(cfg, cfg.input_data)
  image_sample_groups = get_image_sample_groups(data_dir, infer.features)

  # Iterate over the images
  print()
  output_dir = os.path.join(cfg.output_dir, cfg.input_data)
  metric_sum = {metric : 0. for metric in cfg.metric}
  metric_count = 0

  # Saves an image in different formats
  def save_images(path, image, image_srgb, feature_ext=infer.main_feature):
    if feature_ext == 'sh1':
      # Iterate over x, y, z
      for i, axis in [(0, 'x'), (3, 'y'), (6, 'z')]:
        save_images(path, image[:, i:i+3, ...], image_srgb[:, i:i+3, ...], 'sh1' + axis)
      return

    image      = tensor_to_image(image)
    image_srgb = tensor_to_image(image_srgb)
    filename_prefix = path + '.' + feature_ext + '.'
    for format in cfg.format:
      if format in {'exr', 'pfm', 'phm', 'hdr'}:
        # Transform to original range
        if infer.main_feature in {'sh1', 'nrm'}:
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
        target = load_image_features(os.path.join(data_dir, target_name), infer.main_feature)
        target = image_to_tensor(target, batch=True).to(device)
        target_srgb = transform_feature(target, infer.main_feature, 'srgb', tonemap_exposure)

      # Iterate over the input images
      for input_name in input_names:
        print(input_name, '...', end='', flush=True)

        # Load the input image
        input = load_image_features(os.path.join(data_dir, input_name), infer.features)

        # Compute the autoexposure value
        exposure = autoexposure(input) if infer.main_feature == 'hdr' else 1.

        # Infer
        input = image_to_tensor(input, batch=True).to(device)
        output = infer(input, exposure)

        input = input[:, 0:infer.num_main_channels, ...] # keep only the main feature
        input_srgb  = transform_feature(input,  infer.main_feature, 'srgb', tonemap_exposure)
        output_srgb = transform_feature(output, infer.main_feature, 'srgb', tonemap_exposure)

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
        output_suffix = cfg.result if cfg.output_suffix is None else cfg.output_suffix
        output_name = input_name + '.' + output_suffix
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