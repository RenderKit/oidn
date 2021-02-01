## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import sys
import struct
import json
import csv
import zipfile
import time
import socket
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

WORKER_RANK = 0
WORKER_UID = '%08x.%s.%s' % (int(time.time()), socket.gethostname(), os.getpid())

def round_down(a, b):
  return a // b * b

def round_up(a, b):
  return (a + b - 1) // b * b

def round_nearest(a, b):
  return (a + b//2) // b * b

# Prints an error message and exits
def error(*args):
  if WORKER_RANK == 0:
    print('Error:', *args)
  exit(1)

# Prints a warning message
def warning(*args):
  if WORKER_RANK == 0:
    print('Warning:', *args)

# Returns the extension of a path without the dot
def get_path_ext(path):
  return os.path.splitext(path)[1][1:]

## -----------------------------------------------------------------------------
## File I/O
## -----------------------------------------------------------------------------

# Loads an object from a JSON file
def load_json(filename):
  with open(filename, 'r') as f:
    return json.load(f)

# Saves an object into a JSON file
def save_json(filename, obj):
  with open(filename, 'w') as f:
    json.dump(obj, f, indent='\t')

# Loads a list of rows from a CSV file
def load_csv(filename):
  rows = []
  with open(filename) as f:
    csv_reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
      if len(row) == 1:
        rows.append(row[0])
      else:
        rows.append(row)
  return rows

# Saves a list of rows into a CSV file
def save_csv(filename, rows):
  with open(filename, 'w') as f:
    csv_writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    for row in rows:
      if type(row) in [list, tuple]:
        csv_writer.writerow(row)
      else:
        csv_writer.writerow([row])

# Saves a ZIP file containing the specified input files
def save_zip(output_filename, input_filenames, root_dir=None):
  with zipfile.ZipFile(output_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zip:
    for input_filename in input_filenames:
      arcname = os.path.relpath(input_filename, root_dir) if root_dir else None
      zip.write(input_filename, arcname=arcname)

## -----------------------------------------------------------------------------
## PyTorch utils
## -----------------------------------------------------------------------------

# Starts worker processes
def start_workers(cfg, worker_fn):
  if cfg.num_devices > 1:
    # Spawn a worker process for each device
    mp.spawn(worker_fn, args=(cfg,), nprocs=cfg.num_devices)
  else:
    worker_fn(0, cfg)

# Initializes a worker process and returns whether running in distributed mode
def init_worker(rank, cfg):
  if cfg.num_devices > 1:
    # Set 'fork' multiprocessing start method for improved DataLoader performance
    # We must set it explicitly as spawned processes have the 'spawn' method set by default
    mp.set_start_method('fork', force=True)

    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='nccl',
                            rank=rank, world_size=cfg.num_devices,
                            init_method='env://')

    # Set the global rank for this worker process
    global WORKER_RANK
    WORKER_RANK = rank

    # Running in distributed mode
    return True
  else:
    # This is the only worker, not running in distributed mode
    return False

# Cleans up resources used by the worker process
def cleanup_worker(cfg):
  if cfg.num_devices > 1:
    dist.destroy_process_group()

# Initializes and returns the PyTorch device with the specified ID
def init_device(cfg, id=0):
  if cfg.device == 'cuda':
    device = torch.device(cfg.device, id)

    if cfg.deterministic:
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True
    else:
      torch.backends.cudnn.benchmark = True # higher performance

    torch.cuda.set_device(id)
    device_name = torch.cuda.get_device_name()
  elif cfg.device == 'cpu':
    device = torch.device(cfg.device)
    device_name = 'CPU'
  else:
    error('invalid device')

  print(f'Device {id:2}:', device_name)
  return device

# Remove wrappers like DataParallel from a module
def unwrap_module(module):
  if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.DistributedDataParallel):
    return module.module
  else:
    return module

# Generates a random float in [0, 1)
def rand():
  return torch.rand(1).item()

# Generates a random integer in [low, high) or [0, low)
def randint(low, high=None):
  if high is None:
    return torch.randint(low, (1,)).item()
  else:
    return torch.randint(low, high, (1,)).item()

# Generates a random permutation of integers in [0, n)
def randperm(n):
  return torch.randperm(n).tolist()

## -----------------------------------------------------------------------------
## Progress
## -----------------------------------------------------------------------------

# Simple progress bar
class ProgressBar(object):
  def __init__(self, total, prefix='Progress:', width=50):
    self.total = total
    self.prefix = prefix
    self.width = width
    self._start()

  def _start(self):
    self._percent = -100
    self._fill = -1
    self._finished = False
    self.update(0)
    return self

  def update(self, value):
    if self._finished:
      return
    self.value = value
    ratio = float(value) / self.total
    percent = 100 * ratio
    fill = int(round(self.width * ratio))
    if (int(round(percent*10)) != int(round(self._percent*10))) or (fill != self._fill):
      self._percent = percent
      self._fill = fill
      bar = '=' * fill + '.' * (self.width - fill)
      sys.stdout.write('\r%s [%s] %5.1f%%' % (self.prefix, bar, percent))
      sys.stdout.flush()

  def next(self):
    self.update(self.value + 1)

  def finish(self, suffix=''):
    self.update(self.total)
    if suffix:
      sys.stdout.write('\r' + ' ' * (len(self.prefix) + self.width + 10))
      sys.stdout.write('\r%s %s' % (self.prefix, suffix))
    sys.stdout.write('\n')
    sys.stdout.flush()
    self._finished = True

# Formats time (days, hours, minutes, seconds) specified in seconds
def format_time(seconds, precision=None):
  if seconds < 0:
    raise ValueError('seconds must be >= 0')

  units = [('d', 24*60*60), ('h', 60*60), ('m', 60), ('s', 1)]

  num = 0
  text = ''
  total = seconds

  for name, div in units:
    cur = total // div
    if cur > 0 or (num == 0 and div == 1):
      num += 1
      if text: text += ' '
      text += '%d%s' % (cur, name)
      total -= cur * div
    elif num > 0:
      num += 1
    if num == precision:
      return format_time(round_nearest(seconds, div))

  return text