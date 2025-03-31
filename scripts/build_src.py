#!/usr/bin/env python3

## Copyright 2022 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import re
import shutil

from common import *

# Detect the version
print('Detecting the version')
version_file = os.path.join(root_dir, 'README.md')
with open(version_file, 'rb') as f:
  version_text = str(f.read())
version = re.findall('v[0-9a-z.-]+', version_text)[0][1:]

# Copy the source into a temporary directory
print('Copying the source code')
src_name = f'oidn-{version}'
src_dir = os.path.join(root_dir, src_name)
shutil.copytree(root_dir, src_dir, ignore=shutil.ignore_patterns('.git', '.gitmodules', '__pycache__'))

# Create the package
build_dir = os.path.join(root_dir, 'build')
if not os.path.isdir(build_dir):
  os.mkdir(build_dir)
package_filename = os.path.join(build_dir, src_name + '.src' + ('.zip' if OS == 'windows' else '.tar.gz'))
create_package(package_filename, src_dir)

# Remove the temporary directory
shutil.rmtree(src_dir)