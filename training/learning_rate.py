## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import math

# Cyclical learning rate (CLR) with optional linear ramp-down
def get_cyclic_lr_with_ramp_down_function(base_lr,
                                          max_lr,
                                          step_size,
                                          mode='triangular',
                                          gamma=1.,
                                          total_iterations=None,
                                          ramp_down_factor=100):

  def triangular_scale(iterations, cycle):
    return 1

  def triangular2_scale(iterations, cycle):
    return 0.5 ** (cycle - 1)

  def exp_range_scale(iterations, cycle):
    return gamma ** iterations

  if mode == 'triangular':
    scale_fn = triangular_scale
  elif mode == 'triangular2':
    scale_fn = triangular2_scale
  elif mode == 'exp_range':
    scale_fn = exp_range_scale
  else:
    raise ValueError('invalid mode')

  cycle_size = 2 * step_size
  if total_iterations:
    ramp_down_iterations = total_iterations % cycle_size
    cyclic_iterations = total_iterations - ramp_down_iterations

  def get_lr(iterations):
    if not total_iterations or iterations < cyclic_iterations:
      # Cyclical phase
      cycle = math.floor(1 + float(iterations) / cycle_size)
      x = abs(float(iterations) / step_size - 2 * cycle + 1)
      scale = scale_fn(iterations, cycle)
      return base_lr + (max_lr - base_lr) * max(1 - x, 0) * scale
    else:
      # Ramp-down phase
      x = min(float(iterations - cyclic_iterations) / max(ramp_down_iterations, 1), 1)
      return base_lr * (1 - x + x / ramp_down_factor)

  return get_lr