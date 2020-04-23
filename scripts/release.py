#!/usr/bin/env python

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

from __future__ import print_function
import sys
import os
import platform
from glob import glob
import shutil
import tarfile
from zipfile import ZipFile
import re
import argparse

if sys.version_info[0] >= 3:
  from urllib.request import urlretrieve
else:
  from urllib import urlretrieve

MSVC_GENERATOR = 'Visual Studio 15 2017 Win64'
ICC_TOOLCHAIN  = 'Intel C++ Compiler 18.0'
TBB_VERSION    = '2020.1'

def run(command):
  if os.system(command) != 0:
    raise Exception('non-zero return value')

def extract_package(filename, output_dir):
  if re.search(r'^https?://', filename):
    print('Downloading file:', filename)
    file = urlretrieve(filename)[0]
  else:
    file = filename

  print('Extracting package:', filename)
  if re.search(r'(\.tar(\..+)?|tgz)$', filename):
    package = tarfile.open(file)
  elif filename.endswith('.zip'):
    package = ZipFile(file)
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
  compilers = {'Windows' : ['msvc', 'icc'],
               'Linux'   : ['gcc', 'clang', 'icc'],
               'Darwin'  : ['clang', 'icc']}

  parser = argparse.ArgumentParser()
  parser.usage = '\rIntel(R) Open Image Denoise - Release\n' + parser.format_usage()
  parser.add_argument('stage', type=str, nargs='*', choices=['build', 'package'], default='build')
  parser.add_argument('--compiler', type=str, choices=compilers[system], default='icc')
  parser.add_argument('--config', type=str, choices=['Debug', 'Release', 'RelWithDebInfo'], default='Release')
  cfg = parser.parse_args()

  # Set the directories
  root_dir = os.getcwd()
  deps_dir = os.path.join(root_dir, 'deps')
  build_dir = os.path.join(root_dir, 'build_' + cfg.config.lower())

  # Build
  if 'build' in cfg.stage:
    # Set up TBB
    tbb_platform = {'Linux' : 'lin', 'Darwin' : 'mac', 'Windows' : 'win'}[system]
    tbb_dir = os.path.join(deps_dir, 'tbb', TBB_VERSION, tbb_platform)
    if not os.path.isdir(tbb_dir):
      # Download and extract TBB
      tbb_url = 'https://github.com/oneapi-src/oneTBB/releases/download/v%s/tbb-%s-%s' % (TBB_VERSION, TBB_VERSION, tbb_platform)
      tbb_url += '.zip' if system == 'Windows' else '.tgz'
      os.makedirs(tbb_dir)
      extract_package(tbb_url, tbb_dir)
    tbb_root = os.path.join(tbb_dir, 'tbb')

    # Create a clean build directory
    if os.path.isdir(build_dir):
      shutil.rmtree(build_dir)
    os.mkdir(build_dir)
    os.chdir(build_dir)

    if system == 'Windows':
      # Configure
      toolchain = ICC_TOOLCHAIN if cfg.compiler == 'icc' else ''
      run('cmake -L ' +
          '-G "%s" ' % MSVC_GENERATOR +
          '-T "%s" ' % toolchain +
          '-D TBB_ROOT="%s" ' % tbb_root +
          '..')

      # Build
      run('cmake --build . --config %s --target ALL_BUILD' % cfg.config)
    else:
      # Configure
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
    os.chdir(build_dir)

    # Configure
    run('cmake -L -D OIDN_ZIP_MODE=ON ..')

    # Build
    if system == 'Windows':
      run('cmake --build . --config %s --target PACKAGE' % cfg.config)
    else:
      run('cmake --build . --target package -j -v')

    # Extract the package
    package_filename = glob(os.path.join(build_dir, 'oidn-*'))[0]
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
  