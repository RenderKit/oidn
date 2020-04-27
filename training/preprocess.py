#!/usr/bin/env python3

## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import shutil
import numpy as np
import torch

from config import *
from util import *
from dataset import *
from model import *
from color import *
import tza

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Preprocesses training and validation datasets.')

  # Initialize the PyTorch device
  device = init_device(cfg)

  # Initialize the transfer function
  transfer = get_transfer_function(cfg.transfer)

  # Returns a preprocessed image (also changes the original image!)
  def preprocess_image(image, exposure):
    color = image[..., 0:3]
    color = torch.from_numpy(color).to(device)
    if 'hdr' in cfg.features:
      color *= exposure
    color = transfer.forward(color)
    color = torch.clamp(color, max=1.)
    color = color.cpu().numpy()
    image[..., 0:3] = color
    return np.nan_to_num(image.astype(np.float16))

  # Preprocesses a group of input and target images at different SPPs
  def preprocess_sample_group(input_dir, output_tza, input_names, target_name):
    samples = []

    # Load the target image
    print(target_name)
    target_image = load_target_image(os.path.join(input_dir, target_name), cfg.features)

    # Compute the autoexposure value
    exposure = autoexposure(target_image) if 'hdr' in cfg.features else 1.

    # Preprocess the target image
    target_image = preprocess_image(target_image, exposure)

    # Save the target image
    output_tza.write(target_name, target_image, 'hwc')

    # Process the input images
    for input_name in input_names:
      # Load the image
      print(input_name)
      input_image = load_input_image(os.path.join(input_dir, input_name), cfg.features)

      if input_image.shape[0:2] != target_image.shape[0:2]:
        error('the input and target images have different sizes')

      # Preprocess the image
      input_image = preprocess_image(input_image, exposure)

      # Save the image
      output_tza.write(input_name, input_image, 'hwc')

      # Add sample
      samples.append((input_name, target_name))

    return samples

  # Preprocesses a dataset
  def preprocess_dataset(data_name):
    input_dir = get_data_dir(cfg, data_name)
    output_dir = get_preproc_data_dir(cfg, data_name)
    print('Preprocessing dataset:', input_dir)

    # Check whether the dataset exists
    if not os.path.isdir(input_dir):
      print('Does not exist')
      return

    # Create the output directory if it doesn't exist
    if os.path.isdir(output_dir):
      if cfg.clean:
        shutil.rmtree(output_dir)
      else:
        print('Skipping, already preprocessed')
        return
    os.makedirs(output_dir)

    # Save the config
    save_config(output_dir, cfg)

    # Preprocess image sample groups
    sample_groups = get_image_sample_groups(input_dir, cfg.features)
    tza_filename = os.path.join(output_dir, 'images.tza')
    samples = []
    with tza.Writer(tza_filename) as output_tza:
      for _, input_names, target_name in sample_groups:
        if target_name:
          samples += preprocess_sample_group(input_dir, output_tza, input_names, target_name)

    # Save the samples in the dataset
    samples_filename = os.path.join(output_dir, 'samples.json')
    save_json(samples_filename, samples)

  # Preprocess all datasets
  with torch.no_grad():
    preprocess_dataset(cfg.train_data)
    print()
    preprocess_dataset(cfg.valid_data)

if __name__ == '__main__':
  main()