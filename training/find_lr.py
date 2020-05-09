#!/usr/bin/env python3

## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import *
from dataset import *
from model import *
from loss import *
from learning_rate import *
from result import *
from util import *

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Tool for finding the optimal minimum and maximum learning rates.')

  # Generate a result name if not specified
  if not cfg.result:
    cfg.result = '%x' % int(time.time())

  # Set the learning rate range
  cfg.lr     = 1e-8
  cfg.max_lr = 1.

  # Initialize the PyTorch device
  device = init_device(cfg)

  # Initialize the model
  model = UNet(get_num_channels(cfg.features))
  if cfg.device == 'cuda' and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
  model.to(device)

  # Initialize the loss function
  criterion = get_loss_function(cfg.loss)
  criterion.to(device)

  # Initialize the optimizer
  optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

  # Start training
  print('Result:', cfg.result)
  step = 0

  # Initialize the training dataset
  train_data = TrainingDataset(cfg, cfg.train_data)
  if len(train_data) > 0:
    print('Training images:', train_data.num_images)
  else:
    error('no training images')
  pin_memory = (cfg.device != 'cpu')
  train_data_loader = DataLoader(
    train_data, cfg.batch_size, shuffle=True,
    num_workers=cfg.loaders, pin_memory=pin_memory,
    drop_last=True)
  train_steps_per_epoch = len(train_data_loader)

  # Initialize the learning rate scheduler
  gamma = (cfg.max_lr / cfg.lr) ** (1. / (train_steps_per_epoch-1))
  lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

  # Training loop
  start_time = time.time()
  avg_loss = 0.
  best_loss = float('inf')
  beta = 0.98 # loss smoothing
  result = [['learning_rate', 'smoothed_loss', 'loss']]
  progress = ProgressBar(train_steps_per_epoch, 'Train')
  model.train()

  # Iterate over the batches
  for _, batch in enumerate(train_data_loader, 0):
    # Get the batch
    input, target = batch
    input  = input.to(device, non_blocking=True).float()
    target = target.to(device, non_blocking=True).float()

    # Run a training step
    optimizer.zero_grad()
    loss = criterion(model(input), target)
    loss.backward()

    # Compute the smoothed loss
    cur_loss = loss.item()
    avg_loss = beta * avg_loss + (1 - beta) * cur_loss
    smoothed_loss = avg_loss / (1 - beta ** (step+1))
    if smoothed_loss < best_loss:
      best_loss = smoothed_loss
    elif smoothed_loss > 4 * best_loss:
      break

    # Record result
    lr = lr_scheduler.get_last_lr()[0]
    result.append([lr, smoothed_loss, cur_loss])

    # Next step
    optimizer.step()
    lr_scheduler.step()
    step += 1
    progress.next()

  # Print stats
  duration = time.time() - start_time
  images_per_sec = (step * cfg.batch_size) / duration
  progress.finish('(%.1f images/s, %.1fs)' % (images_per_sec, duration))

  # Save the results
  result_filename = os.path.join(cfg.results_dir, cfg.result) + '_lr.csv'
  save_csv(result_filename, result)

if __name__ == '__main__':
  main()