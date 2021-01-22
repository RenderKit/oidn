## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from util import *

def get_model(cfg):
  type = cfg.model
  num_input_channels = len(get_model_channels(cfg.features))
  if type == 'unet':
    return UNet(num_input_channels)
  else:
    error('invalid model')

## -----------------------------------------------------------------------------
## Network layers
## -----------------------------------------------------------------------------

# 3x3 convolution module
def Conv(in_channels, out_channels):
  return nn.Conv2d(in_channels, out_channels, 3, padding=1)

# ReLU function
def relu(x):
  return F.relu(x, inplace=True)

# 2x2 max pool function
def pool(x):
  return F.max_pool2d(x, 2, 2)

# 2x2 nearest-neighbor upsample function
def upsample(x):
  return F.interpolate(x, scale_factor=2, mode='nearest')

# Channel concatenation function
def concat(a, b):
  return torch.cat((a, b), 1)

## -----------------------------------------------------------------------------
## U-Net model
## -----------------------------------------------------------------------------

class UNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=3):
    super(UNet, self).__init__()

    # Number of channels per layer
    ic   = in_channels
    ec1  = 32
    ec2  = 48
    ec3  = 64
    ec4  = 80
    ec5  = 96
    dc4  = 112
    dc3  = 96
    dc2  = 64
    dc1a = 64
    dc1b = 32
    oc   = out_channels

    # Convolutions
    self.enc_conv0  = Conv(ic,      ec1)
    self.enc_conv1  = Conv(ec1,     ec1)
    self.enc_conv2  = Conv(ec1,     ec2)
    self.enc_conv3  = Conv(ec2,     ec3)
    self.enc_conv4  = Conv(ec3,     ec4)
    self.enc_conv5a = Conv(ec4,     ec5)
    self.enc_conv5b = Conv(ec5,     ec5)
    self.dec_conv4a = Conv(ec5+ec3, dc4)
    self.dec_conv4b = Conv(dc4,     dc4)
    self.dec_conv3a = Conv(dc4+ec2, dc3)
    self.dec_conv3b = Conv(dc3,     dc3)
    self.dec_conv2a = Conv(dc3+ec1, dc2)
    self.dec_conv2b = Conv(dc2,     dc2)
    self.dec_conv1a = Conv(dc2+ic,  dc1a)
    self.dec_conv1b = Conv(dc1a,    dc1b)
    self.dec_conv0  = Conv(dc1b,    oc)

    # Images must be padded to multiples of the alignment
    self.alignment = 16

  def forward(self, input):
    # Encoder
    # -------------------------------------------

    x = relu(self.enc_conv0(input))  # enc_conv0

    x = relu(self.enc_conv1(x))      # enc_conv1
    x = pool1 = pool(x)              # pool1

    x = relu(self.enc_conv2(x))      # enc_conv2
    x = pool2 = pool(x)              # pool2

    x = relu(self.enc_conv3(x))      # enc_conv3
    x = pool3 = pool(x)              # pool3

    x = relu(self.enc_conv4(x))      # enc_conv4
    x = pool(x)                      # pool4

    # Bottleneck
    x = relu(self.enc_conv5a(x))     # enc_conv5a
    x = relu(self.enc_conv5b(x))     # enc_conv5b

    # Decoder
    # -------------------------------------------

    x = upsample(x)                  # upsample4
    x = concat(x, pool3)             # concat4
    x = relu(self.dec_conv4a(x))     # dec_conv4a
    x = relu(self.dec_conv4b(x))     # dec_conv4b

    x = upsample(x)                  # upsample3
    x = concat(x, pool2)             # concat3
    x = relu(self.dec_conv3a(x))     # dec_conv3a
    x = relu(self.dec_conv3b(x))     # dec_conv3b

    x = upsample(x)                  # upsample2
    x = concat(x, pool1)             # concat2
    x = relu(self.dec_conv2a(x))     # dec_conv2a
    x = relu(self.dec_conv2b(x))     # dec_conv2b

    x = upsample(x)                  # upsample1
    x = concat(x, input)             # concat1
    x = relu(self.dec_conv1a(x))     # dec_conv1a
    x = relu(self.dec_conv1b(x))     # dec_conv1b

    x = self.dec_conv0(x)            # dec_conv0

    return x