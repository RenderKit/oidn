## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
import time
import torch

from util import *

# Returns the main feature from a list of features
def get_main_feature(features):
  if len(features) > 1:
    features = list(set(features) & {'hdr', 'ldr', 'sh1'})
    if len(features) > 1:
      error('multiple main features specified')
  if not features:
    error('no main feature specified')
  return features[0]

# Returns the auxiliary features from a list of features
def get_aux_features(features):
  main_feature = get_main_feature(features)
  return list(set(features).difference([main_feature]))

# Returns the config filename in a directory
def get_config_filename(dir):
  return os.path.join(dir, 'config.json')

# Loads the config from a directory
def load_config(dir):
  filename = get_config_filename(dir)
  cfg = load_json(filename)
  return argparse.Namespace(**cfg)

# Saves the config to a directory
def save_config(dir, cfg):
  filename = get_config_filename(dir)
  save_json(filename, vars(cfg))

# Parses the config from the command line arguments
def parse_args(cmd=None, description=None):
  def get_default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

  if cmd is None:
    cmd, _ = os.path.splitext(os.path.basename(sys.argv[0]))

  parser = argparse.ArgumentParser(description=description)
  parser.usage = '\rIntel(R) Open Image Denoise - Training\n' + parser.format_usage()
  advanced = parser.add_argument_group('optional advanced arguments')

  parser.add_argument('--config', '-c', type=str, help='load configuration from JSON file (overrides command-line arguments)')

  if cmd in {'preprocess', 'train', 'find_lr'}:
    parser.add_argument('features', type=str, nargs='*',
                        choices=['hdr', 'ldr', 'sh1', 'albedo', 'alb', 'normal', 'nrm', []],
                        help='set of input features')
    parser.add_argument('--clean_aux', action='store_true',
                        help='train with noise-free (reference) auxiliary features')
    parser.add_argument('--filter', '-f', type=str,
                        choices=['RT', 'RTLightmap'],
                        help='filter to train (determines some default arguments)')
    parser.add_argument('--preproc_dir', '-P', type=str, default='preproc',
                        help='directory of preprocessed datasets')
    parser.add_argument('--train_data', '-t', type=str,
                        help='name of the training dataset')
    advanced.add_argument('--transfer', '-x', type=str,
                          choices=['linear', 'srgb', 'pu', 'log'],
                          help='transfer function')

  if cmd in {'preprocess', 'train'}:
    parser.add_argument('--valid_data', '-v', type=str,
                        help='name of the validation dataset')

  if cmd in {'preprocess', 'infer'}:
    parser.add_argument('--data_dir', '-D', type=str, default='data',
                        help='directory of datasets (e.g. training, validation, test)')

  if cmd in {'train', 'find_lr', 'infer', 'export', 'visualize'}:
    parser.add_argument('--results_dir', '-R', type=str, default='results',
                        help='directory of training results')
    parser.add_argument('--result', '-r', type=str, required=(not cmd in {'train', 'find_lr'}),
                        help='name of the training result')

  if cmd in {'infer'}:
    parser.add_argument('--aux_results', '-a', type=str, nargs='*', default=[],
                        help='prefilter auxiliary features using the specified training results')

  if cmd in {'train', 'infer', 'export'}:
    parser.add_argument('--num_epochs', '--epochs', '-e', type=int,
                        default=(2000 if cmd == 'train' else None),
                        help='number of training epochs')

  if cmd in {'train'}:
    parser.add_argument('--num_valid_epochs', '--valid_epochs', type=int, default=10,
                        help='perform validation every this many epochs')
    parser.add_argument('--num_save_epochs', '--save_epochs', type=int, default=10,
                        help='save checkpoints every this many epochs')
    parser.add_argument('--lr', '--learning_rate', type=float,
                        help='initial learning rate')
    parser.add_argument('--max_lr', '--max_learning_rate', type=float,
                        help='maximum learning rate')
    parser.add_argument('--lr_warmup', '--learning_rate_warmup', type=float, default=0.15,
                        help='the percentage of the cycle spent increasing the learning rate (warm-up)')

  if cmd in {'find_lr'}:
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-8,
                        help='minimum learning rate')
    parser.add_argument('--max_lr', '--max_learning_rate', type=float, default=0.1,
                        help='maximum learning rate')

  if cmd in {'train', 'find_lr'}:
    parser.add_argument('--batch_size', '--bs', '-b', type=int, default=16,
                        help='mini-batch size (total batch size of all devices)')
    parser.add_argument('--num_loaders', '--loaders', '-j', type=int, default=4,
                        help='number of data loader threads per device')
    parser.add_argument('--precision', '-p', type=str, choices=['fp32', 'mixed'],
                        help='training precision')
    advanced.add_argument('--model', '-m', type=str, choices=['unet'], default='unet',
                          help='network model')
    advanced.add_argument('--loss', '-l', type=str,
                          choices=['l1', 'mape', 'smape', 'l2', 'ssim', 'msssim', 'l1_msssim', 'l1_grad'],
                          default='l1_msssim',
                          help='loss function')
    advanced.add_argument('--msssim_weights', type=float, nargs='*',
                          help='MS-SSIM scale weights')
    advanced.add_argument('--tile_size', '--ts', type=int, default=256,
                          help='size of the cropped image tiles')
    advanced.add_argument('--seed', '-s', type=int,
                          help='seed for random number generation')

  if cmd in {'infer', 'compare_image'}:
    parser.add_argument('--metric', '-M', type=str, nargs='*',
                        choices=['psnr', 'mse', 'ssim', 'msssim'], default=['psnr', 'ssim'],
                        help='metrics to compute')

  if cmd in {'infer'}:
    parser.add_argument('--input_data', '-i', type=str, default='test',
                        help='name of the input dataset')
    parser.add_argument('--output_dir', '-O', type=str, default='infer',
                        help='directory of output images')
    parser.add_argument('--output_suffix', '-o', type=str,
                        help='suffix of the output image names')
    parser.add_argument('--format', '-F', type=str, nargs='*', default=['exr'],
                        help='output image formats')
    parser.add_argument('--save_all', action='store_true',
                        help='save input and target images too')

  if cmd in {'export'}:
    parser.add_argument('target', type=str, nargs='?',
                        choices=['weights', 'package'], default='weights',
                        help='what to export')
    parser.add_argument('--output', '-o', type=str,
                        help='output file')

  if cmd in {'convert_image', 'split_exr'}:
    parser.add_argument('input', type=str,
                        help='input image')

  if cmd in {'compare_image'}:
    parser.add_argument('input', type=str, nargs=2,
                        help='input images')

  if cmd in {'convert_image'}:
    parser.add_argument('output', type=str,
                        help='output image')

  if cmd in {'convert_image', 'compare_image'}:
    parser.add_argument('--exposure', '-E', type=float, default=1.,
                        help='linear exposure scale for HDR image')

  if cmd in {'split_exr'}:
    parser.add_argument('--layer', type=str,
                        help='name of the image layer')

  if cmd in {'preprocess', 'train', 'find_lr', 'infer', 'export'}:
    parser.add_argument('--device', '-d', type=str,
                        choices=['cpu', 'cuda'], default=get_default_device(),
                        help='type of device(s) to use')
    parser.add_argument('--device_id', '-k', type=int, default=0,
                        help='ID of the first device to use')
    parser.add_argument('--num_devices', '-n', type=int, default=1,
                        help='number of devices to use (with IDs device_id .. device_id+num_devices-1)')
    advanced.add_argument('--deterministic', '--det', action='store_true',
                          default=(cmd in {'preprocess', 'infer', 'export'}),
                          help='makes computations deterministic (slower performance)')

  cfg = parser.parse_args()

  # Load and apply configuration from file if specified
  if cfg.config is not None:
    cfg_dict = vars(cfg)
    cfg_dict.update(load_json(cfg.config))
    cfg = argparse.Namespace(**cfg_dict)

  if cmd in {'preprocess', 'train', 'find_lr'}:
    # Check the filter
    if cfg.filter is None:
      warning('filter not specified, using generic default arguments')

    # Replace feature names with IDs
    FEATURE_IDS = {'albedo' : 'alb', 'normal' : 'nrm'}
    cfg.features = [FEATURE_IDS.get(f, f) for f in cfg.features]
    # Remove duplicate features
    cfg.features = list(dict.fromkeys(cfg.features).keys())

    # Set the default transfer function
    if cfg.transfer is None:
      main_feature = get_main_feature(cfg.features)
      if main_feature == 'hdr':
        cfg.transfer = 'log' if cfg.filter == 'RTLightmap' else 'pu'
      elif main_feature in {'ldr', 'alb'}:
        cfg.transfer = 'srgb'
      else:
        cfg.transfer = 'linear'

    # Set the default datasets
    if cfg.train_data is None and (cmd == 'find_lr' or cfg.valid_data is None):
      cfg.train_data = 'train'
      if cmd != 'find_lr':
        cfg.valid_data = 'valid'

  if cmd in {'train', 'find_lr'}:
    # Check the batch size
    if cfg.batch_size % cfg.num_devices != 0:
      parser.error('batch_size is not divisible by num_devices')

    # Set the default result name (generated)
    if cfg.result is None:
      cfg.result = WORKER_UID

    # Set the default MS-SSIM weights
    if cfg.msssim_weights is None:
      if cfg.filter == 'RT':
        cfg.msssim_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

  if cmd in {'train'}:
    # Set the default training precision
    if cfg.precision is None:
      cfg.precision = 'mixed' if cfg.device == 'cuda' else 'fp32'
      
    # Set the default maximum learning rate
    if cfg.max_lr is None:
      cfg.max_lr = 3.125e-6 * cfg.batch_size

  # Print PyTorch version
  print('PyTorch:', torch.__version__)

  return cfg