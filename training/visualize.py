#!/usr/bin/env python

## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os

from config import *
from util import *
from result import *

# Parse the command line arguments
cfg = parse_args(description='Invokes TensorBoard for visualizing statistics for a training result.')

result_dir = get_result_dir(cfg)
if not os.path.isdir(result_dir):
  error('result does not exist')

# Run TensorBoard
log_dir = os.path.join(result_dir, 'log')
os.system('tensorboard --logdir=' + log_dir)

