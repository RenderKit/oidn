#!/usr/bin/env python

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
from array import array

def write_prologue(out_file, in_name):
  out_file.write('// Generated from: %s\n\n' % in_name)

def write_namespace_begin(out_file, scopes):
  for s in scopes:
    out_file.write('namespace %s {\n' % s)
  if scopes:
    out_file.write('\n')

def write_namespace_end(out_file, scopes):
  if scopes:
    out_file.write('\n')
  for scope in reversed(scopes):
    out_file.write('} // namespace %s\n' % scope)

def generate(in_path, cpp_path, hpp_path, namespace):
  scopes = []
  if namespace:
    scopes = namespace.split('::')

  in_name  = os.path.basename(in_path)
  var_name = os.path.splitext(in_name)[0]

  # Read the input file
  with open(in_path, 'rb') as in_file:
    in_data = array('B', in_file.read())
  in_size = len(in_data)

  # Write the source file
  with open(cpp_path, 'w') as cpp_file:
    write_prologue(cpp_file, in_name)
    write_namespace_begin(cpp_file, scopes)

    cpp_file.write('unsigned char %s[%d] = {' % (var_name, in_size))
    for i in range(in_size):
      c = in_data[i]
      if i > 0:
        cpp_file.write(',')
      if (i+1) % 20 == 1:
        cpp_file.write('\n')
      cpp_file.write('%d' % c)
    cpp_file.write('\n};\n')

    write_namespace_end(cpp_file, scopes)

  # Write the header file
  if hpp_path:
    with open(hpp_path, 'w') as hpp_file:
      write_prologue(hpp_file, in_name)
      write_namespace_begin(hpp_file, scopes)
      hpp_file.write('extern unsigned char %s[%d];\n' % (var_name, in_size))
      write_namespace_end(hpp_file, scopes)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generates C++ source/header file(s) from a binary file.')
  parser.add_argument('-n', '--namespace', help='C++ namespace to use')
  parser.add_argument('-o', '--output', required=True, help='output C++ source filename')
  parser.add_argument('-H', '--output-header', help='output C++ header filename')
  parser.add_argument('input')
  args = parser.parse_args()

  in_path = args.input
  if not os.path.exists(in_path):
    parser.error('input file does not exist')

  generate(in_path, args.output, args.output_header, args.namespace)