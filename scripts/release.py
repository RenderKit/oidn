#!/usr/bin/env python

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

from __future__ import print_function
import os
import platform
from glob import glob
import shutil
import tarfile
import zipfile
import re
import argparse

TBB_VERSION = '2020.1'

def run(command):
  if os.system(command) != 0:
    raise Exception('non-zero return value')

def extract_package(filename, output_dir):
  print('Extracting package:', filename)
  if re.search(r'\.tar(\..*)?$', filename):
    package = tarfile.open(filename)
  elif filename.endswith('.zip'):
    package = zipfile.open(filename)
  else:
    raise Exception('unsupported package format')
  package.extractall(output_dir)
  package.close()

def check_symbols(filename, label, max_version):
  with os.popen("nm \"%s\" | tr ' ' '\n' | grep @@%s_" % (filename, label)) as out:
    for line in out:
      symbol = line.strip()
      _, version = symbol.split('@@')
      _, version = version.split('_')
      version = [int(v) for v in version.split('.')]
      if version > list(max_version):
        raise Exception('problematic symbol %s in %s' % (symbol, os.path.basename(filename)))

def check_symbols_linux(filename):
  print('Checking symbols:', filename)
  check_symbols(filename, 'GLIBC',   (2, 17, 0))
  check_symbols(filename, 'GLIBCXX', (3, 4, 19))
  check_symbols(filename, 'CXXABI',  (1, 3, 7))

def main():
  # Detect the OS
  system = platform.system() # Linux, Darwin, Windows

  # Parse the arguments
  parser = argparse.ArgumentParser()
  parser.usage = '\rIntel(R) Open Image Denoise - Release\n' + parser.format_usage()
  parser.add_argument('stage', type=str, nargs='*', choices=['build', 'package'], default='build')
  parser.add_argument('--compiler', type=str, choices=['gcc', 'clang', 'icc'], default='icc')
  parser.add_argument('--config', type=str, choices=['Debug', 'Release', 'RelWithDebInfo'], default='Release')
  cfg = parser.parse_args()

  # Set the directories
  root_dir = os.getcwd()
  deps_dir = os.path.join(root_dir, 'deps')
  build_dir = os.path.join(root_dir, 'build_' + cfg.config.lower())

  # Build
  if 'build' in cfg.stage:
    # Set up TBB
    tbb_dir = os.path.join(deps_dir, 'tbb', TBB_VERSION)
    tbb_platform = {'Linux' : 'linux', 'Darwin' : 'mac', 'Windows' : 'win'}[system]
    tbb_root = os.path.join(tbb_dir, tbb_platform, 'tbb')
    if not os.path.isdir(tbb_root):
      raise Exception('cannot find TBB root at %s. Download TBB using scripts/download_tbb.sh.' % tbb_root)

    # Create a clean build directory
    if os.path.isdir(build_dir):
      shutil.rmtree(build_dir)
    os.mkdir(build_dir)

    # Configure
    os.chdir(build_dir)
    cc = cfg.compiler
    cxx = {'gcc' : 'g++', 'clang' : 'clang++', 'icc' : 'icpc'}[cc]
    run('cmake -L ' +
        '-D CMAKE_C_COMPILER:FILEPATH=%s ' % cc +
        '-D CMAKE_CXX_COMPILER:FILEPATH=%s ' % cxx +
        '-D CMAKE_BUILD_TYPE=%s ' % cfg.config +
        '-D TBB_ROOT="%s" ' % tbb_root +
        '..')

    # Build
    run('cmake --build . --target preinstall -j -v')
    
  # Package
  if 'package' in cfg.stage:
    # Configure
    os.chdir(build_dir)
    run('cmake -L ' +
        '-D OIDN_ZIP_MODE=ON ' +
        '..')

    # Build
    run('cmake --build . --target package -j -v')

    # Extract the package
    package_filename = glob(os.path.join(build_dir, 'oidn-*.x86_64.*'))[0]
    extract_package(package_filename, build_dir)
    package_dir = re.sub(r'\.(tar(\..*)?|zip)$', '', package_filename)

    # Get the list of binaries
    if system == 'Windows':
      binaries  = glob(os.path.join(package_dir, 'bin', '*.exe'))
      binaries += glob(os.path.join(package_dir, 'bin', '*.dll'))
    else:
      binaries  = glob(os.path.join(package_dir, 'bin', '*'))
      binaries += glob(os.path.join(package_dir, 'lib', '*.so*'))
    binaries = list(filter(lambda f: os.path.isfile(f) and not os.path.islink(f), binaries))

    # Check the symbols in the binaries
    if system == 'Linux':
      for f in binaries:
        check_symbols_linux(f)

    # Delete the extracted package
    shutil.rmtree(package_dir)

if __name__ == '__main__':
  main()
  