## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import platform

# Runs a command and checks the return value for success
def run(command):
  status = os.system(command)
  if status != 0:
    print('Error: non-zero return value')
    exit(1)

# Detect the OS and architecture
OS = {'Windows' : 'windows', 'Linux' : 'linux', 'Darwin' : 'macos'}[platform.system()]
ARCH = platform.machine()

# Get the root directory
root_dir = os.environ.get('OIDN_ROOT_DIR')
if root_dir is None:
  root_dir = os.getcwd()