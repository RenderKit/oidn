## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import *
from image import *
from ssim import SSIM, MS_SSIM

# L1 loss (seems to be faster than the built-in L1Loss)
class L1Loss(nn.Module):
  def forward(self, input, target):
    return torch.abs(input - target).mean()

# L2 (MSE) loss
class L2Loss(nn.Module):
  def forward(self, input, target):
    return ((input - target) ** 2).mean()

# MAPE (relative L1) loss
class MAPELoss(nn.Module):
  def forward(self, input, target):
    return (torch.abs(input - target) / (torch.abs(target) + 1e-2)).mean()

# SMAPE (symmetric MAPE) loss
class SMAPELoss(nn.Module):
  def forward(self, input, target):
    return (torch.abs(input - target) / (torch.abs(input) + torch.abs(target) + 1e-2)).mean()

# SSIM loss
class SSIMLoss(nn.Module):
  def __init__(self):
    super(SSIMLoss, self).__init__()
    self.ssim = SSIM(data_range=1.)

  def forward(self, input, target):
    return 1. - self.ssim(input, target)

# MS-SSIM loss
class MSSSIMLoss(nn.Module):
  def __init__(self):
    super(MSSSIMLoss, self).__init__()
    self.msssim = MS_SSIM(data_range=1.)

  def forward(self, input, target):
    return 1. - self.msssim(input, target)

# Gradient loss
class GradientLoss(nn.Module):
  def forward(self, input, target):
    return torch.abs(gradient(input) - gradient(target)).mean()

# Mix loss
class MixLoss(nn.Module):
  def __init__(self, loss1, loss2, alpha):
    super(MixLoss, self).__init__()
    self.loss1 = loss1
    self.loss2 = loss2
    self.alpha = alpha

  def forward(self, input, target):
    return (1. - self.alpha) * self.loss1(input, target) + self.alpha * self.loss2(input, target)

def get_loss_function(type):
  if type == 'l1':
    return L1Loss()
  elif type == 'l2':
    return L2Loss()
  elif type == 'mape':
    return MAPELoss()
  elif type == 'smape':
    return SMAPELoss()
  elif type == 'ssim':
    return SSIMLoss()
  elif type == 'msssim':
    return MSSSIMLoss()
  elif type == 'l1_msssim':
    # [Zhao et al., 2018, "Loss Functions for Image Restoration with Neural Networks"]
    return MixLoss(L1Loss(), MSSSIMLoss(), 0.84)
  elif type == 'l1_grad':
    return MixLoss(L1Loss(), GradientLoss(), 0.5)
  else:
    error('invalid loss function')
