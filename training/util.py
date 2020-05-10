## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import sys
import struct
import json
import csv
from zipfile import ZipFile
import numpy as np
import torch
import torch.nn as nn

def round_down(a, b):
  return a // b * b

def round_up(a, b):
  return (a + b - 1) // b * b

def round_nearest(a, b):
  return (a + b//2) // b * b

# Prints an error message and exits
def error(*args):
  print('Error:', *args)
  exit(1)

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
def save_zip(filename, input_filenames):
  with ZipFile(filename, 'w') as zip:
    for input_filename in input_filenames:
      zip.write(input_filename)

## -----------------------------------------------------------------------------
## PyTorch utils
## -----------------------------------------------------------------------------

# Initializes and returns the PyTorch device
def init_device(cfg):
  print('PyTorch:', torch.__version__)

  # Query CPU information
  #num_sockets = int(os.popen("lscpu -b -p=Socket | grep -v '^#' | sort -u | wc -l").read())
  num_cores    = int(os.popen("lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l").read())

  # Configure OpenMP
  os.environ['OMP_NUM_THREADS'] = str(num_cores)
  os.environ['KMP_BLOCKTIME']   = '0'
  os.environ['KMP_AFFINITY']    = 'granularity=fine,compact,1,0'

  # Initialize the device
  device = torch.device(cfg.device)
  if cfg.device == 'cuda':
    if cfg.deterministic:
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True
    else:
      torch.backends.cudnn.benchmark = True # higher performance
    device_name = torch.cuda.get_device_name()
  else:
    device_name = 'CPU'
  print('Device:', device_name)
  return device

# Remove wrappers like DataParallel from a module
def unwrap_module(module):
  if isinstance(module, nn.DataParallel):
    return module.module
  else:
    return module
  
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