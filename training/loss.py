## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from util import *
from image import *
from ssim import SSIM, MS_SSIM

def get_loss_function(cfg):
  num_channels = len(get_model_channels(get_main_feature(cfg.features)))
  type = cfg.loss

  if type == 'l1':
    return L1Loss()
  elif type == 'l2':
    return L2Loss()
  elif type == 'mape':
    return MAPELoss()
  elif type == 'smape':
    return SMAPELoss()
  elif type == 'ssim':
    return SSIMLoss(num_channels=num_channels)
  elif type == 'msssim':
    return MSSSIMLoss(num_channels=num_channels, weights=cfg.msssim_weights)
  elif type == 'l1_msssim':
    # [Zhao et al., 2018, "Loss Functions for Image Restoration with Neural Networks"]
    msssim_loss = MSSSIMLoss(num_channels=num_channels, weights=cfg.msssim_weights)
    return MixLoss([L1Loss(), msssim_loss], [0.16, 0.84])
  elif type == 'l1_grad':
    return MixLoss([L1Loss(), GradientLoss()], [0.5, 0.5])
  else:
    error('invalid loss function')

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
  def __init__(self, num_channels=3):
    super(SSIMLoss, self).__init__()
    self.ssim = SSIM(data_range=1., channel=num_channels)

  def forward(self, input, target):
    with torch.autocast('cuda', enabled=False):
      return 1. - self.ssim(input.float(), target.float())

# MS-SSIM loss
class MSSSIMLoss(nn.Module):
  def __init__(self, num_channels=3, weights=None):
    super(MSSSIMLoss, self).__init__()
    self.msssim = MS_SSIM(data_range=1., channel=num_channels, weights=weights)

  def forward(self, input, target):
    with torch.autocast('cuda', enabled=False):
      return 1. - self.msssim(input.float(), target.float())

# Gradient loss
class GradientLoss(nn.Module):
  def forward(self, input, target):
    return torch.abs(tensor_gradient(input) - tensor_gradient(target)).mean()

# Mix loss
class MixLoss(nn.Module):
  def __init__(self, losses, weights):
    super(MixLoss, self).__init__()
    self.losses  = nn.Sequential(*losses)
    self.weights = weights

  def forward(self, input, target):
    return sum([l(input, target) * w for l, w in zip(self.losses, self.weights)])