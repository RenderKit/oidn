## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import torch

from util import *

# Gets the path to the result directory
def get_result_dir(cfg, result=None):
  if result is None:
    result = cfg.result
  return os.path.join(cfg.results_dir, result)

# Gets the path to the result checkpoint directory
def get_checkpoint_dir(result_dir):
  return os.path.join(result_dir, 'checkpoints')

# Gets the path to a checkpoint file
def get_checkpoint_filename(result_dir, epoch):
  checkpoint_dir = get_checkpoint_dir(result_dir)
  return os.path.join(checkpoint_dir, 'checkpoint_%d.pth' % epoch)

# Gets the path to the file that contains the checkpoint state (latest epoch)
def get_checkpoint_state_filename(result_dir):
  checkpoint_dir = get_checkpoint_dir(result_dir)
  return os.path.join(checkpoint_dir, 'latest')

# Gets the latest checkpoint epoch
def get_latest_checkpoint_epoch(result_dir):
  latest_filename = get_checkpoint_state_filename(result_dir)
  if not os.path.isfile(latest_filename):
    error('no checkpoints found')
  with open(latest_filename, 'r') as f:
    return int(f.readline())

# Gets the path to result log directory
def get_result_log_dir(result_dir):
  return os.path.join(result_dir, 'log')

# Saves a training checkpoint
def save_checkpoint(result_dir, epoch, step, model, optimizer):
  checkpoint_dir = get_checkpoint_dir(result_dir)
  if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  checkpoint_filename = get_checkpoint_filename(result_dir, epoch)
  torch.save({
               'epoch': epoch,
               'step': step,
               'model_state': unwrap_module(model).state_dict(),
               'optimizer_state': optimizer.state_dict(),
             }, checkpoint_filename)

  latest_filename = get_checkpoint_state_filename(result_dir)
  with open(latest_filename, 'w') as f:
    f.write('%d' % epoch)

# Loads and returns a training checkpoint
def load_checkpoint(result_dir, device, epoch=None, model=None, optimizer=None):
  if epoch is None or epoch <= 0:
    epoch = get_latest_checkpoint_epoch(result_dir)

  checkpoint_filename = get_checkpoint_filename(result_dir, epoch)
  if not os.path.isfile(checkpoint_filename):
    error('checkpoint does not exist')

  checkpoint = torch.load(checkpoint_filename, map_location=device)

  if checkpoint['epoch'] != epoch:
    error('checkpoint epoch mismatch')
  if model:
    unwrap_module(model).load_state_dict(checkpoint['model_state'])
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer_state'])

  return checkpoint