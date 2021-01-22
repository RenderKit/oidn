#!/usr/bin/env python3

## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os

from config import *
from util import *
from result import *

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Invokes TensorBoard for visualizing statistics of a training result.')

  result_dir = get_result_dir(cfg)
  if not os.path.isdir(result_dir):
    error('result does not exist')

  # Run TensorBoard
  log_dir = os.path.join(result_dir, 'log')
  os.system('tensorboard --logdir=' + log_dir)

if __name__ == '__main__':
  main()
