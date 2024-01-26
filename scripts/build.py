#!/usr/bin/env python3

## Copyright 2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import re
import sys
import os
from glob import glob
import shutil
import argparse

from common import *

ISPC_VERSION = '1.21.0' if OS == 'windows' and ARCH == 'arm64' else '1.22.0'
TBB_VERSION  = '2021.11.0'

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
  check_symbols(filename, 'GLIBC',   (2, 28, 0))
  check_symbols(filename, 'GLIBCXX', (3, 4, 22))
  check_symbols(filename, 'CXXABI',  (1, 3, 11))

# Parse the arguments
compilers = {'windows' : ['msvc17', 'msvc16-icc21', 'msvc16-icc20', 'msvc16-icc19', 'msvc16', 'msvc15-icc20', 'msvc15-icc19', 'msvc15-icc18', 'msvc15', 'clang'],
             'linux'   : ['gcc', 'clang', 'icc'],
             'macos'   : ['clang', 'icc']}

parser = argparse.ArgumentParser()
parser.usage = '\rIntel(R) Open Image Denoise - Build\n' + parser.format_usage()
parser.add_argument('target', type=str, nargs='?', choices=['all', 'install', 'package'], default='all')
parser.add_argument('--build_dir', '-B', type=str, help='build directory')
parser.add_argument('--install_dir', '-I', type=str, help='install directory')
parser.add_argument('--compiler', type=str, choices=(['default'] + compilers[OS]), default='default')
parser.add_argument('--config', type=str, choices=['Debug', 'Release', 'RelWithDebInfo'], default='Release')
parser.add_argument('--full', action='store_true', help='build with full device support')
parser.add_argument('--wrapper', type=str, help='wrap build command')
parser.add_argument('-D', dest='cmake_vars', type=str, action='append', help='create or update a CMake cache entry')
cfg = parser.parse_args()

if cfg.build_dir is None:
  cfg.build_dir = os.path.join(root_dir, 'build')
else:
  cfg.build_dir = os.path.abspath(cfg.build_dir)

if cfg.install_dir is None:
  cfg.install_dir = os.path.join(root_dir, 'install')
else:
  cfg.install_dir = os.path.abspath(cfg.install_dir)

if cfg.compiler == 'default' and cfg.full:
  cfg.compiler = 'clang'

# Create a clean build directory
if os.path.isdir(cfg.build_dir):
  shutil.rmtree(cfg.build_dir)
os.mkdir(cfg.build_dir)
os.chdir(cfg.build_dir)

# Configure
msbuild = False
config_cmd = 'cmake -L'

if OS == 'windows':
  if cfg.compiler == 'clang':
    cc  = 'clang'
    cxx = 'clang++'
    config_cmd += ' -G Ninja'
    config_cmd += f' -D CMAKE_C_COMPILER:FILEPATH="{cc}"'
    config_cmd += f' -D CMAKE_CXX_COMPILER:FILEPATH="{cxx}"'
  else:
    msbuild = True
    if cfg.compiler == 'default':
      cfg.compiler = 'msvc17'
    for compiler in cfg.compiler.split('-'):
      if compiler.startswith('msvc'):
        msvc_arch = {'x86_64': 'x64', 'arm64': 'ARM64'}[ARCH]
        config_cmd += {'msvc15' :  ' -G "Visual Studio 15 2017 Win64"',
                       'msvc16' : f' -G "Visual Studio 16 2019" -A {msvc_arch}',
                       'msvc17' : f' -G "Visual Studio 17 2022" -A {msvc_arch}'}[compiler]
      elif compiler.startswith('icc'):
        icc_version = {'18' : '18.0', '19' : '19.0', '20' : '19.1', '21' : '19.2'}[compiler[3:]]
        config_cmd += f' -T "Intel C++ Compiler {icc_version}"'
else:
  if OS == 'linux':
    config_cmd += ' -G Ninja'
  if cfg.compiler != 'default':
    cc = cfg.compiler
    cxx = {'gcc' : 'g++', 'clang' : 'clang++', 'icx' : 'icx', 'icc' : 'icpc'}[cc]
    if cfg.compiler == 'icc':
      icc_dir = os.environ.get('OIDN_ICC_DIR_' + OS.upper())
      if icc_dir:
        cc  = os.path.join(icc_dir, cc)
        cxx = os.path.join(icc_dir, cxx)
    config_cmd += f' -D CMAKE_C_COMPILER:FILEPATH="{cc}"'
    config_cmd += f' -D CMAKE_CXX_COMPILER:FILEPATH="{cxx}"'

# Set up the dependencies
deps_dir = os.path.join(root_dir, 'deps')
if not os.path.isdir(deps_dir):
  os.makedirs(deps_dir)

# Set up TBB
tbb_release = f'oneapi-tbb-{TBB_VERSION}-'
tbb_release += {'windows' : 'win', 'linux' : 'lin', 'macos' : 'mac'}[OS]
tbb_dir = os.path.join(deps_dir, tbb_release)
if OS == 'macos':
  tbb_dir += f'.{ARCH}'
tbb_root = os.path.join(tbb_dir, f'oneapi-tbb-{TBB_VERSION}')
if not os.path.isdir(tbb_dir):
  if OS != 'macos' and ARCH != 'arm64':
    # Download and extract TBB
    tbb_url = f'https://github.com/oneapi-src/oneTBB/releases/download/v{TBB_VERSION}/{tbb_release}'
    tbb_url += '.zip' if OS == 'windows' else '.tgz'
    tbb_filename = download_file(tbb_url, deps_dir)
    extract_package(tbb_filename, tbb_dir)
    os.remove(tbb_filename)
  else:
    # Download TBB source
    tbb_url = f'https://github.com/oneapi-src/oneTBB/archive/refs/tags/v{TBB_VERSION}.tar.gz'
    tbb_filename = download_file(tbb_url, deps_dir)
    extract_package(tbb_filename, deps_dir)
    os.remove(tbb_filename)

    # Build TBB
    tbb_src_dir = os.path.join(deps_dir, f'oneTBB-{TBB_VERSION}')
    tbb_build_dir = os.path.join(tbb_src_dir, 'build')
    os.mkdir(tbb_build_dir)
    os.chdir(tbb_build_dir)
    tbb_config_cmd = config_cmd + f' -D CMAKE_BUILD_TYPE=Release -D TBB_TEST=OFF -D CMAKE_INSTALL_PREFIX={tbb_root} ..'
    if OS == 'macos':
      min_macos_version = {'x86_64' : '10.11', 'arm64' : '11.0'}[ARCH]
      tbb_config_cmd += f' -D CMAKE_OSX_DEPLOYMENT_TARGET={min_macos_version}'
    run(tbb_config_cmd)
    if msbuild:
      run('cmake --build . --config Release --target INSTALL')
    else:
      run('cmake --build . --target install')
    os.chdir(cfg.build_dir)
    shutil.rmtree(tbb_src_dir)
config_cmd += f' -D TBB_ROOT="{tbb_root}"'

# Set up ISPC
ispc_release = f'ispc-v{ISPC_VERSION}-'
ispc_release += {'windows' : 'windows', 'linux' : 'linux', 'macos' : 'macOS.universal'}[OS]
if OS == 'linux' and ARCH == 'arm64':
  ispc_release += '.aarch64'
ispc_dir = os.path.join(deps_dir, ispc_release)
if not os.path.isdir(ispc_dir):
  # Download and extract ISPC
  ispc_url = f'https://github.com/ispc/ispc/releases/download/v{ISPC_VERSION}/{ispc_release}'
  ispc_url += '.zip' if OS == 'windows' else '.tar.gz'
  ispc_filename = download_file(ispc_url, deps_dir)
  extract_package(ispc_filename, ispc_dir)
  os.remove(ispc_filename)
ispc_executable = os.path.join(ispc_dir, 'bin', 'ispc')
if OS == 'windows':
  ispc_executable += '.exe'
config_cmd += f' -D ISPC_EXECUTABLE="{ispc_executable}"'

config_cmd += f' -D CMAKE_BUILD_TYPE={cfg.config}'

if cfg.full:
  if OS != 'macos':
    config_cmd += ' -D OIDN_DEVICE_CPU=ON -D OIDN_DEVICE_SYCL=ON -D OIDN_DEVICE_CUDA=ON -D OIDN_DEVICE_HIP=ON'
  elif ARCH == 'arm64':
    config_cmd += ' -D OIDN_DEVICE_CPU=ON -D OIDN_DEVICE_METAL=ON'

config_cmd += ' -D OIDN_WARN_AS_ERRORS=ON'

if cfg.target in {'install', 'package'}:
  config_cmd += ' -D OIDN_INSTALL_DEPENDENCIES=ON'

if cfg.target == 'package':
  config_cmd += ' -D OIDN_ZIP_MODE=ON'

if cfg.target == 'install':
  config_cmd += f' -D CMAKE_INSTALL_PREFIX={cfg.install_dir}'

if cfg.cmake_vars:
  for var in cfg.cmake_vars:
    config_cmd += f' -D {var}'

config_cmd += ' ..'

run(config_cmd)

# Build
build_cmd  = 'cmake --build .'

if msbuild:
  cmake_target = {'all' : 'ALL_BUILD', 'install' : 'INSTALL', 'package' : 'PACKAGE'}[cfg.target]
  build_cmd += f' --config {cfg.config} --target {cmake_target}'
else:
  build_cmd += f' --target {cfg.target}'
  if OS == 'macos':
    build_cmd += ' -- VERBOSE=1' # Make
  else:
    build_cmd += ' -- -v' # Ninja

if cfg.wrapper:
  build_cmd = cfg.wrapper + ' ' + build_cmd
run(build_cmd)

if cfg.target == 'package':
  # Extract the package
  package_filename = [f for f in glob(os.path.join(cfg.build_dir, 'oidn-*')) if os.path.isfile(f)][0]
  package_dir = re.sub(r'\.(tar(\..*)?|zip)$', '', package_filename)
  if os.path.isdir(package_dir):
    shutil.rmtree(package_dir)
  extract_package(package_filename, cfg.build_dir)

  # Get the list of binaries
  binaries = glob(os.path.join(package_dir, 'bin', '*'))
  if OS == 'linux':
    binaries += glob(os.path.join(package_dir, 'lib', '*.so*'))
  elif OS == 'macos':
    binaries += glob(os.path.join(package_dir, 'lib', '*.dylib'))
  binaries = list(filter(lambda f: os.path.isfile(f) and not os.path.islink(f), binaries))

  # Check the symbols in the binaries
  if OS == 'linux':
    for filename in binaries:
      check_symbols_linux(filename)

  # Sign the binaries
  sign_file_env = 'SIGN_FILE_' + {'windows' : 'WINDOWS', 'linux' : 'LINUX_disabled', 'macos' : 'MAC'}[OS]
  sign_file = os.environ.get(sign_file_env)
  if sign_file:
    for filename in binaries:
      run(f'{sign_file} -q -vv {filename}')

  # Make the binaries consistently executable
  if OS != 'windows':
    for filename in binaries:
      run(f'chmod +x {filename}')

  # Repack
  if sign_file or OS != 'windows':
    os.remove(package_filename)
    create_package(package_filename, package_dir)

  # Delete the extracted package
  shutil.rmtree(package_dir)