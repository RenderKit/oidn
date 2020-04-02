## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import torch

from util import *

# Gets the path to the result directory
def get_result_dir(cfg):
  return os.path.join(cfg.results_dir, cfg.result)

# Gets the latest chechpoint epoch
def get_latest_checkpoint_epoch(cfg):
  checkpoint_dir = os.path.join(get_result_dir(cfg), 'checkpoints')
  latest_filename = os.path.join(checkpoint_dir, 'latest')
  if not os.path.isfile(latest_filename):
    error('no checkpoints found')
  with open(latest_filename, 'r') as f:
    return int(f.readline())

# Saves a training checkpoint
def save_checkpoint(cfg, epoch, step, model, optimizer):
  checkpoint_dir = os.path.join(get_result_dir(cfg), 'checkpoints')
  if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  checkpoint_filename = os.path.join(checkpoint_dir, 'checkpoint_%d.pth' % epoch)
  torch.save({
               'epoch': epoch,
               'step': step,
               'model_state': unwrap_module(model).state_dict(),
               'optimizer_state': optimizer.state_dict(),
             }, checkpoint_filename)

  latest_filename = os.path.join(checkpoint_dir, 'latest')
  with open(latest_filename, 'w') as f:
    f.write('%d' % epoch)

# Loads and returns a training checkpoint
def load_checkpoint(cfg, device, epoch=None, model=None, optimizer=None):
  if not epoch or epoch <= 0:
    epoch = get_latest_checkpoint_epoch(cfg)

  checkpoint_dir = os.path.join(get_result_dir(cfg), 'checkpoints')
  checkpoint_filename = os.path.join(checkpoint_dir, 'checkpoint_%d.pth' % epoch)
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