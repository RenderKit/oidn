#!/usr/bin/env python

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
from glob import glob
from shutil import which
import argparse

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Compares images produced by the library with generated baseline images.')
parser.usage = '\rIntel(R) Open Image Denoise - Regression Test\n' + parser.format_usage()
parser.add_argument('command', type=str, nargs='*', choices=['generate', 'test'], help='tasks to perform')
parser.add_argument('--filter', '-f', nargs='*', choices=['RT', 'RTLightmap'], default=None, help='filters to test')
parser.add_argument('--build_dir', '-B', type=str, default='build', help='build directory')
parser.add_argument('--data_dir', '-D', type=str, default=os.path.join('training', 'data'), help='directory of datasets (e.g. training, validation, test)')
parser.add_argument('--results_dir', '-R', type=str, default=os.path.join('training', 'results'), help='directory of training results')
parser.add_argument('--baseline_dir', '-G', type=str, default=os.path.join('training', 'infer'), help='directory of generated baseline images')
parser.add_argument('--arch', '-a', type=str, nargs='*', choices=['native', 'pnr', 'hsw', 'skx', 'knl'], help='CPU architectures to test')
parser.add_argument('--log', '-l', type=str, default='regression.log', help='output log file')
cfg = parser.parse_args()

if not cfg.arch:
  cfg.arch = ['native']
  # Detect whether Intel(R) Software Development Emulator (SDE) is installed
  # See: https://software.intel.com/en-us/articles/intel-software-development-emulator
  if which('sde'):
    cfg.arch += ['pnr', 'hsw', 'skx', 'knl'] # Penryn, Haswell, Skylake-X, Knights Landing

# Runs tests for the specified model
def test(result, filter, features, dataset):
  # Generate baseline images
  if 'generate' in cfg.command:
    print('Generate:', result)
    infer_cmd = os.path.join('training', 'infer.py')
    infer_cmd += ' -D "%s" -R "%s" -O "%s" -i %s -r %s -F exr -d cpu' % (cfg.data_dir, cfg.results_dir, cfg.baseline_dir, dataset, result)
    
    os.system('echo "%s" >> %s' % (infer_cmd, cfg.log))
    infer_cmd += ' >> %s' % cfg.log

    if os.system(infer_cmd) != 0:
      print('Error: inference failed')
      exit(1)

  if 'test' in cfg.command:
    main_feature = features[0]

    # Gather the list of images
    dataset_dir = os.path.join(cfg.data_dir, dataset)
    image_filenames = sorted(glob(os.path.join(dataset_dir, '**', '*.%s.exr' % main_feature), recursive=True))
    image_names = [os.path.relpath(filename, dataset_dir).rsplit('.', 3)[0] for filename in image_filenames]

    # Iterate over architectures
    for arch in cfg.arch:
      # Iterate over the images
      for image_name in image_names:
        print('Test:', result, arch, image_name)
        denoise_cmd = os.path.join(cfg.build_dir, 'denoise')

        ref_filename = os.path.join(cfg.baseline_dir, dataset, '%s_%s.%s.exr' % (image_name, result, main_feature))
        if not os.path.isfile(ref_filename):
          print('Error: missing baseline image (run with "generate" first)')
          exit(1)
        denoise_cmd += ' -f %s -v 2 --ref %s' % (filter, ref_filename)

        for feature in features:
          feature_filename = os.path.join(dataset_dir, image_name) + '.%s.exr' % feature
          denoise_cmd += ' --%s %s' % (feature, feature_filename)

        if arch != 'native':
          denoise_cmd = ('sde -%s -- ' % arch) + denoise_cmd

        os.system('echo >> %s' % cfg.log)
        os.system('echo "%s" >> %s' % (denoise_cmd, cfg.log))
        denoise_cmd += ' >> %s' % cfg.log

        if os.system(denoise_cmd) != 0:
          exit(1)

# Filter: RT
if not cfg.filter or 'RT' in cfg.filter:
  dataset = 'rt_test'
  test('rt_hdr_alb_nrm', 'RT', ['hdr', 'alb', 'nrm'], dataset)
  test('rt_hdr_alb',     'RT', ['hdr', 'alb'],        dataset)
  test('rt_hdr',         'RT', ['hdr'],               dataset)
  test('rt_ldr_alb_nrm', 'RT', ['ldr', 'alb', 'nrm'], dataset)
  test('rt_ldr_alb',     'RT', ['ldr', 'alb'],        dataset)
  test('rt_ldr',         'RT', ['ldr'],               dataset)

# Filter: RTLightmap
if not cfg.filter or 'RTLightmap' in cfg.filter:
  dataset = 'rtlightmap_test'
  test('rtlightmap_hdr', 'RTLightmap', ['hdr'], dataset)

if 'test' in cfg.command:
  print('Success: all tests passed')