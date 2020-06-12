#!/usr/bin/env python3

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import sys
import shutil
from glob import glob
from shutil import which
import argparse

from common import *

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Runs all tests, including comparing images produced by the library with generated baseline images.')
parser.usage = '\rIntel(R) Open Image Denoise - Test\n' + parser.format_usage()
parser.add_argument('command', type=str, nargs='?', choices=['generate', 'run'], default='run')
parser.add_argument('--filter', '-f', type=str, nargs='*', choices=['RT', 'RTLightmap'], default=None, help='filters to test')
parser.add_argument('--build_dir', '-B', type=str, help='build directory')
parser.add_argument('--data_dir', '-D', type=str, default=os.path.join(root_dir, 'training', 'data'), help='directory of datasets (e.g. training, validation, test)')
parser.add_argument('--results_dir', '-R', type=str, default=os.path.join(root_dir, 'training', 'results'), help='directory of training results')
parser.add_argument('--baseline_dir', '-G', type=str, help='directory of generated baseline images')
parser.add_argument('--arch', '-a', type=str, nargs='*', choices=['native', 'pnr', 'hsw', 'skx', 'knl'], help='CPU architectures to test')
parser.add_argument('--log', '-l', type=str, default=os.path.join(root_dir, 'test.log'), help='output log file')
cfg = parser.parse_args()

if cfg.baseline_dir is None:
  cfg.baseline_dir = os.environ.get('OIDN_BASELINE_DIR_' + OS.upper())
  if cfg.baseline_dir is None:
    cfg.baseline_dir = os.path.join(root_dir, 'training', 'infer')

if cfg.command == 'run':
  # Detect the binary directory
  if cfg.build_dir is None:
    cfg.build_dir = os.path.join(root_dir, 'build')
  else:
    cfg.build_dir = os.path.abspath(cfg.build_dir)

  bin_dir = os.path.join(cfg.build_dir, 'install', 'bin')
  if not os.path.isdir(bin_dir):
    bin_dir = os.path.join(root_dir, 'build')

  # Detect whether Intel(R) Software Development Emulator (SDE) is installed
  # See: https://software.intel.com/en-us/articles/intel-software-development-emulator
  sde_dir = os.environ.get('OIDN_SDE_DIR_' + OS.upper())
  sde = os.path.join(sde_dir, 'sde') if sde_dir else 'sde'
  if cfg.arch is None:
    cfg.arch = ['native']
    if shutil.which(sde):
      cfg.arch += ['pnr', 'hsw', 'skx', 'knl'] # Penryn, Haswell, Skylake-X, Knights Landing

# Runs main tests
def test():
  if cfg.command == 'run':
    # Iterate over architectures
    for arch in cfg.arch:
      # Run test
      test_name = arch
      print('Test:', test_name, '...')

      test_cmd = os.path.join(bin_dir, 'oidnTest')
      if arch != 'native':
        test_cmd = f'{sde} -{arch} -- ' + test_cmd

      if os.system(test_cmd) != 0:
        exit(1)

# Runs regression tests for the specified filter
def test_regression(filter, features, dataset):
  # Get the result name
  result = filter.lower()
  for f in features:
    result += '_' + f
  features_str = result.split('_', 1)[1]

  dataset_dir = os.path.join(cfg.data_dir, dataset)

  if cfg.command == 'generate':
    # Generate baseline images
    gen_name = f'{filter}.{features_str}'
    print('Generate:', gen_name, '...')

    # Convert the input images to PFM
    image_filenames = sorted(glob(os.path.join(dataset_dir, '**', '*.exr'), recursive=True))
    for input_filename in image_filenames:
      output_filename = input_filename.rsplit('.', 1)[0] + '.pfm'
      convert_cmd = os.path.join(root_dir, 'training', 'convert_image.py')
      convert_cmd += f' "{input_filename}" "{output_filename}"'
      run(f'echo "{convert_cmd}" >> {cfg.log}')
      run(convert_cmd)

    # Run inference for the input images
    infer_cmd = os.path.join(root_dir, 'training', 'infer.py')
    infer_cmd += f' -D "{cfg.data_dir}" -R "{cfg.results_dir}" -O "{cfg.baseline_dir}" -i {dataset} -r {result} -F pfm -d cpu'
    run(f'echo "{infer_cmd}" >> {cfg.log}')
    infer_cmd += f' >> {cfg.log}'
    run(infer_cmd)

  elif cfg.command == 'run':
    main_feature = features[0]

    # Gather the list of images
    image_filenames = sorted(glob(os.path.join(dataset_dir, '**', f'*.{main_feature}.pfm'), recursive=True))
    if not image_filenames:
      print('Error: converted input images missing (run with "generate" first)')
      exit(1)
    image_names = [os.path.relpath(filename, dataset_dir).rsplit('.', 3)[0] for filename in image_filenames]

    # Iterate over architectures
    for arch in cfg.arch:
      # Iterate over images
      for image_name in image_names:
        # Iterate over in-place mode
        for inplace in [False, True]:
          # Iterate over maximum memory usages (tiling)
          for maxmem in [None, 512]:
            # Run test
            test_name = f'{filter}.{features_str}.{arch}.{image_name}'
            if inplace:
              test_name += '.inplace'
            if maxmem:
              test_name += f'.{maxmem}mb'
            print('Test:', test_name, '...')

            denoise_cmd = os.path.join(bin_dir, 'oidnDenoise')

            ref_filename = os.path.join(cfg.baseline_dir, dataset, f'{image_name}.{result}.{main_feature}.pfm')
            if not os.path.isfile(ref_filename):
              print('Error: baseline image missing (run with "generate" first)')
              exit(1)
            denoise_cmd += f' -f {filter} -v 2 --ref "{ref_filename}"'

            for feature in features:
              feature_filename = os.path.join(dataset_dir, image_name) + f'.{feature}.pfm'
              denoise_cmd += f' --{feature} "{feature_filename}"'

            if inplace:
              denoise_cmd += ' --inplace'

            if maxmem:
              denoise_cmd += f' --maxmem {maxmem}'

            if arch != 'native':
              denoise_cmd = f'{sde} -{arch} -- ' + denoise_cmd

            run(f'echo >> "{cfg.log}"')
            run(f'echo "{denoise_cmd}" >> "{cfg.log}"')
            denoise_cmd += f' >> "{cfg.log}"'

            if os.system(denoise_cmd) != 0:
              exit(1)

# Main tests
test()

# Regression tests: RT
if not cfg.filter or 'RT' in cfg.filter:
  dataset = 'rt_regress'
  test_regression('RT', ['hdr', 'alb', 'nrm'], dataset)
  test_regression('RT', ['hdr', 'alb'],        dataset)
  test_regression('RT', ['hdr'],               dataset)
  test_regression('RT', ['ldr', 'alb', 'nrm'], dataset)
  test_regression('RT', ['ldr', 'alb'],        dataset)
  test_regression('RT', ['ldr'],               dataset)

# Regression tests: RTLightmap
if not cfg.filter or 'RTLightmap' in cfg.filter:
  dataset = 'rtlightmap_regress'
  test_regression('RTLightmap', ['hdr'], dataset)

# Done
if cfg.command == 'run':
  print('Success: all tests passed')