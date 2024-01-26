## Copyright 2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import platform
import re
import shutil
import tarfile
from zipfile import ZipFile
from urllib.request import urlretrieve

# Runs a command and checks the return value for success
def run(command):
  status = os.system(command)
  if status != 0:
    print('Error: non-zero return value')
    exit(1)

def download_file(url, output_dir):
  print('Downloading file:', url)
  filename = os.path.join(output_dir, os.path.basename(url))
  urlretrieve(url, filename=filename)
  return filename

def extract_package(filename, output_dir):
  print('Extracting package:', filename)
  # Detect the package format and open the package
  if re.search(r'(\.tar(\..+)?|tgz)$', filename):
    package = tarfile.open(filename)
    members = package.getnames()
  elif filename.endswith('.zip'):
    package = ZipFile(filename)
    members = package.namelist()
  else:
    raise Exception('unsupported package format')
  # Avoid nesting two top-level directories with the same name
  if os.path.commonpath(members) == os.path.basename(output_dir):
    output_dir = os.path.dirname(output_dir)
  # Create the output directory if it doesn't exist
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
  # Extract the package
  package.extractall(output_dir)
  package.close()

def create_package(filename, input_dir):
  print('Creating package:', filename)
  if filename.endswith('.tar.gz'):
    with tarfile.open(filename, "w:gz") as package:
      package.add(input_dir, arcname=os.path.basename(input_dir))
  elif filename.endswith('.zip'):
    shutil.make_archive(filename[:-4], 'zip', os.path.dirname(input_dir), os.path.basename(input_dir))
  else:
    raise Exception('unsupported package format')

# Detect the OS and architecture
OS = {'Windows' : 'windows', 'Linux' : 'linux', 'Darwin' : 'macos'}[platform.system()]

ARCH = platform.machine().lower()
if ARCH == 'amd64':
  ARCH = 'x86_64'
elif ARCH == 'aarch64':
  ARCH = 'arm64'

# Get the root directory
root_dir = os.environ.get('OIDN_ROOT_DIR')
if root_dir is None:
  root_dir = os.getcwd()