#!/usr/bin/env python3

## Copyright 2018 Intel Corporation
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
from infer import Infer
import tza

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Preprocesses training and validation datasets.')
  main_feature = get_main_feature(cfg.features)
  aux_features = get_aux_features(cfg.features)
  all_channels = get_dataset_channels(cfg.features)
  num_main_channels = len(get_dataset_channels(main_feature))

  # Initialize the PyTorch device
  device = init_device(cfg)

  # Initialize the transfer function
  transfer = get_transfer_function(cfg)

  # Initialize auxiliary feature inference
  aux_infers = {}
  for aux_result in set(cfg.aux_results):
    aux_infer = Infer(cfg, device, aux_result, is_aux=True)
    if (aux_infer.main_feature not in aux_features) or aux_infer.aux_features:
      error(f'result {aux_result} does not correspond to an auxiliary feature')
    aux_infers[aux_infer.main_feature] = aux_infer

  # Determine the input and target features
  if cfg.clean_aux:
    input_features  = [main_feature]
    target_features = cfg.features
  else:
    input_features  = cfg.features
    target_features = [main_feature]

  # Returns a preprocessed image (also changes the original image!)
  def preprocess_image(image, exposure, prefilter=False):
    # Apply the transfer function to the main feature
    color = image[..., 0:num_main_channels]
    color = torch.from_numpy(color).to(device)
    if main_feature == 'hdr':
      color *= exposure
    color = transfer.forward(color)
    color = torch.clamp(color, max=1.)
    color = color.cpu().numpy()
    image[..., 0:num_main_channels] = color

    # Prefilter the auxiliary features
    if prefilter:
      for aux_feature, aux_infer in aux_infers.items():
        aux_channels = get_dataset_channels(aux_feature)
        aux_channel_indices = get_channel_indices(aux_channels, all_channels)
        aux = image[..., aux_channel_indices]
        aux = image_to_tensor(aux, batch=True).to(device)
        aux = aux_infer(aux)
        aux = tensor_to_image(aux)
        image[..., aux_channel_indices] = aux

    # Convert to FP16
    return np.nan_to_num(image.astype(np.float16))

  # Preprocesses a group of input and target images at different SPPs
  def preprocess_sample_group(input_dir, output_tza, input_names, target_name):
    samples = []

    # Load the target image
    print(target_name)
    target_image, _ = load_image_features(os.path.join(input_dir, target_name), target_features)

    # Compute the autoexposure value
    exposure = autoexposure(target_image) if main_feature == 'hdr' else 1.

    # Preprocess the target image
    target_image = preprocess_image(target_image, exposure)

    # Save the target image
    output_tza.write(target_name, target_image, 'hwc')

    # Process the input images
    for input_name in input_names:
      # Load the image
      print(input_name)
      input_image, _ = load_image_features(os.path.join(input_dir, input_name), input_features)

      if input_image.shape[0:2] != target_image.shape[0:2]:
        error('the input and target images have different sizes')

      # Preprocess the image
      input_image = preprocess_image(input_image, exposure, prefilter=True)

      # Save the image
      output_tza.write(input_name, input_image, 'hwc')

      # Add sample
      samples.append((input_name, target_name))

    return samples

  # Preprocesses a dataset
  def preprocess_dataset(data_name):
    input_dir = get_data_dir(cfg, data_name)
    print('\nDataset:', input_dir)
    if not os.path.isdir(input_dir):
      print('Not found')
      return

    # Create the output directory
    output_name = data_name + '.' + WORKER_UID
    output_dir = os.path.join(cfg.preproc_dir, output_name)
    os.makedirs(output_dir)

    # Preprocess image sample groups
    sample_groups = get_image_sample_groups(input_dir, input_features, target_features)
    tza_filename = os.path.join(output_dir, 'images.tza')
    samples = []
    with tza.Writer(tza_filename) as output_tza:
      for _, input_names, target_name in sample_groups:
        if target_name:
          samples += preprocess_sample_group(input_dir, output_tza, input_names, target_name)

    # Save the samples in the dataset
    samples_filename = os.path.join(output_dir, 'samples.json')
    save_json(samples_filename, samples)

    # Save the config
    save_config(output_dir, cfg)

  # Preprocess all datasets
  with torch.inference_mode():
    for dataset in [cfg.train_data, cfg.valid_data]:
      if dataset:
        preprocess_dataset(dataset)

if __name__ == '__main__':
  main()