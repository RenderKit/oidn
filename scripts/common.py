## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import platform

def run(command):
  status = os.system(command)
  if status != 0:
    print('Error: non-zero return value')
    exit(1)

# Detect the OS
OS = {'Windows' : 'windows', 'Linux' : 'linux', 'Darwin' : 'macos'}[platform.system()]

# Get the root directory
root_dir = os.environ.get('OIDN_ROOT_DIR')
if not root_dir:
  root_dir = os.getcwd()