#!/usr/bin/env python3

## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import sys
import shutil
from glob import glob
from shutil import which
import argparse

from common import *

MODEL_VERSION='v1.4.0'

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Runs all tests, including comparing images produced by the library with generated baseline images.')
parser.usage = '\rIntel(R) Open Image Denoise - Test\n' + parser.format_usage()
parser.add_argument('command', type=str, nargs='?', choices=['baseline', 'run'], default='run')
parser.add_argument('--filter', '-f', type=str, nargs='*', choices=['RT', 'RTLightmap'], default=None, help='filters to test')
parser.add_argument('--build_dir', '-B', type=str, help='build directory')
parser.add_argument('--data_dir', '-D', type=str, help='directory of datasets (e.g. training, validation, test)')
parser.add_argument('--results_dir', '-R', type=str, help='directory of training results')
parser.add_argument('--baseline_dir', '-G', type=str, help='directory of generated baseline images')
parser.add_argument('--arch', '-a', type=str, nargs='*', choices=['native', 'pnr', 'hsw', 'skx', 'knl'], default=['native'], help='CPU architectures to test (requires Intel SDE)')
parser.add_argument('--log', '-l', type=str, default=os.path.join(root_dir, 'test.log'), help='output log file')
cfg = parser.parse_args()

training_dir = os.environ.get('OIDN_TRAINING_DIR_' + OS.upper())
if training_dir is None:
  training_dir = os.path.join(root_dir, 'training')
if cfg.data_dir is None:
  cfg.data_dir = os.path.join(training_dir, 'data')
if cfg.results_dir is None:
  cfg.results_dir = os.path.join(training_dir, 'results')
if cfg.baseline_dir is None:
  cfg.baseline_dir = os.path.join(training_dir, 'baseline_' + MODEL_VERSION)

if cfg.command == 'run':
  # Detect the OIDN binary directory
  if cfg.build_dir is None:
    cfg.build_dir = os.path.join(root_dir, 'build')
  else:
    cfg.build_dir = os.path.abspath(cfg.build_dir)

  bin_dir = os.path.join(cfg.build_dir, 'install', 'bin')
  if not os.path.isdir(bin_dir):
    bin_dir = os.path.join(root_dir, 'build')

  # Detect the Intel(R) Software Development Emulator (SDE)
  # See: https://software.intel.com/en-us/articles/intel-software-development-emulator
  sde = 'sde.exe' if OS == 'windows' else 'sde64'
  sde_dir = os.environ.get('OIDN_SDE_DIR_' + OS.upper())
  if sde_dir is not None:
    sde = os.path.join(sde_dir, sde)

# Prints the name of a test
def print_test(name, kind='Test'):
  print(kind + ':', name, '...', end='', flush=True)

# Runs a test command
def run_test(cmd, arch='native'):
  # Run test through SDE if required
  if arch != 'native':
    cmd = f'{sde} -{arch} -- ' + cmd

  # Write command and redirect output to log
  run(f'echo >> "{cfg.log}"')
  run(f'echo "{cmd}" >> "{cfg.log}"')
  cmd += f' >> "{cfg.log}" 2>&1'

  # Run the command and check the return value
  if os.system(cmd) == 0:
    print(' PASSED')
  else:
    print(' FAILED')
    print(f'Error: test failed, see "{cfg.log}" for details')
    exit(1)

# Runs main tests
def test():
  if cfg.command == 'run':
    # Iterate over architectures
    for arch in cfg.arch:
      print_test(f'oidnTest.{arch}')
      run_test(os.path.join(bin_dir, 'oidnTest'), arch)

# Gets the option name of a feature
def get_feature_opt(feature):
  if feature == 'calb':
    return 'alb'
  elif feature == 'cnrm':
    return 'nrm'
  else:
    return feature

# Gets the file extension of a feature
def get_feature_ext(feature):
  if feature == 'dir':
    return 'sh1x'
  else:
    return get_feature_opt(feature)

# Runs regression tests for the specified filter
def test_regression(filter, feature_sets, dataset):
  dataset_dir = os.path.join(cfg.data_dir, dataset)

  # Convert the input images to PFM
  if cfg.command == 'baseline':
    image_filenames = sorted(glob(os.path.join(dataset_dir, '**', '*.exr'), recursive=True))
    for input_filename in image_filenames:
      input_name = os.path.relpath(input_filename, dataset_dir).rsplit('.', 1)[0]
      print_test(f'{filter}.{input_name}', 'Convert')

      output_filename = input_filename.rsplit('.', 1)[0] + '.pfm'
      convert_cmd = os.path.join(root_dir, 'training', 'convert_image.py')
      convert_cmd += f' "{input_filename}" "{output_filename}"'
      run_test(convert_cmd)

  # Iterate over the feature sets
  for features, full_test in feature_sets:
    # Get the result name
    result = filter.lower()
    for f in features:
      result += '_' + f
    features_str = result.split('_', 1)[1]

    if cfg.command == 'baseline':
      # Generate the baseline images
      print_test(f'{filter}.{features_str}', 'Infer')
      infer_cmd = os.path.join(root_dir, 'training', 'infer.py')
      infer_cmd += f' -D "{cfg.data_dir}" -R "{cfg.results_dir}" -O "{cfg.baseline_dir}" -i {dataset} -r {result} -F pfm -d cpu'
      run_test(infer_cmd)

    elif cfg.command == 'run':
      main_feature = features[0]
      main_feature_ext = get_feature_ext(main_feature)

      # Gather the list of images
      image_filenames = sorted(glob(os.path.join(dataset_dir, '**', f'*.{main_feature_ext}.pfm'), recursive=True))
      if not image_filenames:
        print('Error: baseline input images missing (run with "baseline" first)')
        exit(1)
      image_names = [os.path.relpath(filename, dataset_dir).rsplit('.', 3)[0] for filename in image_filenames]

      # Iterate over architectures
      for arch in cfg.arch:
        # Iterate over images
        for image_name in image_names:
          # Iterate over in-place mode
          for inplace in ([False, True] if full_test else [False]):
            # Run test
            test_name = f'{filter}.{features_str}.{arch}.{image_name}'
            if inplace:
              test_name += '.inplace'
            print_test(test_name)

            denoise_cmd = os.path.join(bin_dir, 'oidnDenoise')

            ref_filename = os.path.join(cfg.baseline_dir, dataset, f'{image_name}.{result}.{main_feature_ext}.pfm')
            if not os.path.isfile(ref_filename):
              print('Error: baseline output image missing (run with "baseline" first)')
              exit(1)
            denoise_cmd += f' -f {filter} -v 2 --ref "{ref_filename}"'

            for feature in features:
              feature_opt = get_feature_opt(feature)
              feature_ext = get_feature_ext(feature)
              feature_filename = os.path.join(dataset_dir, image_name) + f'.{feature_ext}.pfm'
              denoise_cmd += f' --{feature_opt} "{feature_filename}"'

            if set(features) & {'calb', 'cnrm'}:
              denoise_cmd += ' --clean_aux'

            if inplace:
              denoise_cmd += ' --inplace'

            run_test(denoise_cmd, arch)

# Main tests
test()

# Regression tests: RT
if not cfg.filter or 'RT' in cfg.filter:
  test_regression(
    'RT',
    [
      (['hdr', 'alb', 'nrm'],   True),
      (['hdr', 'alb'],          True),
      (['hdr'],                 True),
      (['hdr', 'calb', 'cnrm'], False),
      (['ldr', 'alb', 'nrm'],   False),
      (['ldr', 'alb'],          False),
      (['ldr'],                 True),
      (['ldr', 'calb', 'cnrm'], False),
      (['alb'],                 True),
      (['nrm'],                 True)
    ],
    'rt_regress'
  )

# Regression tests: RTLightmap
if not cfg.filter or 'RTLightmap' in cfg.filter:
  test_regression(
    'RTLightmap',
    [
      (['hdr'], True),
      (['dir'], True)
    ],
    'rtlightmap_regress'
  )

# Done
if cfg.command == 'run':
  print('Success: all tests passed')