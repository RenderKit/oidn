## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
from glob import glob
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import *
from util import *
from image import *
from color import *
import tza

# Returns the ordered list of channel names for the specified features
def get_channels(features, target):
  assert target in {'dataset', 'model'}
  channels = []
  if 'hdr' in features:
    channels += ['hdr.r', 'hdr.g', 'hdr.b']
  if 'ldr' in features:
    channels += ['ldr.r', 'ldr.g', 'ldr.b']
  if 'sh1' in features:
    if target == 'model':
      channels += ['sh1.r', 'sh1.g', 'sh1.b']
    else:
      channels += ['sh1x.r', 'sh1x.g', 'sh1x.b', 'sh1y.r', 'sh1y.g', 'sh1y.b', 'sh1z.r', 'sh1z.g', 'sh1z.b']
  if 'alb' in features:
    channels += ['alb.r', 'alb.g', 'alb.b']
  if 'nrm' in features:
    channels += ['nrm.x', 'nrm.y', 'nrm.z']
  return channels

def get_dataset_channels(features):
  return get_channels(features, target='dataset')

def get_model_channels(features):
  return get_channels(features, target='model')

# Returns the indices of the specified channels in the list of all channels
def get_channel_indices(channels, all_channels):
  return [all_channels.index(ch) for ch in channels]

# Shuffles channels according to the specified order and optionally keeps only
# the specified amount of channels
def shuffle_channels(channels, first_channel, order, num_channels=None):
  first = channels.index(first_channel)
  new_channels = [channels[first+i] for i in order]
  for i in range(len(new_channels)):
    channels[first+i] = new_channels[i]
  if num_channels is not None:
    del channels[first+num_channels:first+len(new_channels)]

# Checks whether the image with specified features exists
def image_exists(name, features):
  suffixes = features.copy()
  if 'sh1' in suffixes:
    suffixes.remove('sh1')
    suffixes += ['sh1x', 'sh1y', 'sh1z']
  return all([os.path.isfile(name + '.' + s + '.exr') for s in suffixes])

# Returns the feature an image represents given its filename
def get_image_feature(filename):
  filename_split = filename.rsplit('.', 2)
  if len(filename_split) < 2:
    return 'srgb' # no extension, assume sRGB
  else:
    ext = filename_split[-1].lower()
    if ext in {'exr', 'pfm', 'phm', 'hdr'}:
      if len(filename_split) == 3:
        feature = filename_split[-2]
        if feature in {'sh1x', 'sh1y', 'sh1z'}:
          feature = 'sh1'
        return feature
      else:
        return 'hdr' # assume HDR
    else:
      return 'srgb' # assume sRGB

# Loads image features in EXR format with given filename prefix
def load_image_features(name, features):
  images = []

  # HDR color
  if 'hdr' in features:
    hdr = load_image(name + '.hdr.exr', num_channels=3)
    hdr = np.maximum(hdr, 0.)
    images.append(hdr)

  # LDR color
  if 'ldr' in features:
    ldr = load_image(name + '.ldr.exr', num_channels=3)
    ldr = np.clip(ldr, 0., 1.)
    images.append(ldr)

  # SH L1 color coefficients
  if 'sh1' in features:
    sh1x = load_image(name + '.sh1x.exr', num_channels=3)
    sh1y = load_image(name + '.sh1y.exr', num_channels=3)
    sh1z = load_image(name + '.sh1z.exr', num_channels=3)

    for sh1 in [sh1x, sh1y, sh1z]:
      # Clip to [-1..1] range (coefficients are assumed to be normalized)
      sh1 = np.clip(sh1, -1., 1.)

      # Transform to [0..1] range
      sh1 = sh1 * 0.5 + 0.5

      images.append(sh1)

  # Albedo
  if 'alb' in features:
    albedo = load_image(name + '.alb.exr', num_channels=3)
    albedo = np.clip(albedo, 0., 1.)
    images.append(albedo)

  # Normal
  if 'nrm' in features:
    normal = load_image(name + '.nrm.exr', num_channels=3)
    normal = np.clip(normal, -1., 1.)
    
    # Transform to [0..1] range
    normal = normal * 0.5 + 0.5

    images.append(normal)

  # Concatenate all feature images into one image
  return np.concatenate(images, axis=2)

# Tries to load metadata for an image with given filename/prefix, returns None if it fails
def load_image_metadata(name):
  dirname, basename = os.path.split(name)
  basename = basename.split('.')[0] # remove all extensions
  while basename:
    metadata_filename = os.path.join(dirname, basename) + '.json'
    if os.path.isfile(metadata_filename):
      return load_json(metadata_filename)
    if '_' in basename:
      basename = basename.rsplit('_', 1)[0]
    else:
      break
  return None

# Saves image metadata to a file with given prefix
def save_image_metadata(name, metadata):
  save_json(name + '.json', metadata)

# Returns a dataset directory path
def get_data_dir(cfg, name):
  return os.path.join(cfg.data_dir, name)

# Returns groups of image samples (input and target images at different SPPs) as
# a list of (group name, list of input names, target name)
def get_image_sample_groups(dir, input_features, target_features=None):
  image_filenames = glob(os.path.join(dir, '**', '*.*.exr'), recursive=True)
  if target_features is None:
    target_features = [get_main_feature(input_features)]

  # Make image groups
  image_groups = defaultdict(set)
  for filename in image_filenames:
    image_name = os.path.relpath(filename, dir)  # remove dir path
    image_name, _, _ = image_name.rsplit('.', 2) # remove extensions
    group = image_name
    if '_' in image_name:
      prefix, suffix = image_name.rsplit('_', 1)
      suffix = suffix.lower()
      if (suffix.isdecimal() or
          (suffix.endswith('spp') and suffix[:-3].isdecimal()) or
          suffix == 'ref' or suffix == 'reference' or
          suffix == 'gt' or suffix == 'target'):
        group = prefix
    image_groups[group].add(image_name)

  # Make sorted image sample (inputs + target) groups
  image_sample_groups = []
  for group in sorted(image_groups):
    # Get the list of inputs and the target
    image_names = sorted(image_groups[group])
    if len(image_names) > 1:
      input_names, target_name = image_names[:-1], image_names[-1]
    else:
      input_names, target_name = image_names, None

    # Check whether all required features exist
    if all([image_exists(os.path.join(dir, input_name), input_features) for input_name in input_names]):
      if target_name and not image_exists(os.path.join(dir, target_name), target_features):
        target_name = None # discard target due to missing features

      # Add sample
      image_sample_groups.append((group, input_names, target_name))

  return image_sample_groups

# Transforms a feature image to another feature type
def transform_feature(image, input_feature, output_feature, exposure=1.):
  if input_feature == 'hdr' and output_feature in {'ldr', 'srgb'}:
    image = tonemap(image * exposure)
  if output_feature == 'srgb':
    if input_feature in {'hdr', 'ldr', 'alb'}:
      image = srgb_forward(image)
    elif input_feature in {'nrm', 'sh1'}:
      # Transform [-1, 1] -> [0, 1]
      image = image * 0.5 + 0.5
  return image

# Returns a data loader and its sampler for the specified dataset
def get_data_loader(rank, cfg, dataset, shuffle=False):
  if cfg.num_devices > 1:
    sampler = DistributedSampler(dataset,
                                 num_replicas=cfg.num_devices,
                                 rank=rank,
                                 shuffle=shuffle)
  else:
    sampler = None

  loader = DataLoader(dataset,
                      batch_size=(cfg.batch_size // cfg.num_devices),
                      sampler=sampler,
                      shuffle=(shuffle if sampler is None else False),
                      num_workers=cfg.num_loaders,
                      pin_memory=(cfg.device != 'cpu'))

  return loader, sampler

## -----------------------------------------------------------------------------
## Preprocessed dataset
## -----------------------------------------------------------------------------

# Returns the directory path of the best matching preprocessed dataset
def get_preproc_data_dir(cfg, name):
  # Get all preprocessed versions of the requested dataset
  data_dirs = sorted([f for f in glob(os.path.join(cfg.preproc_dir, name + '.*')) if os.path.isdir(f)])

  # Iterate over all dataset versions
  best_dir = None
  best_num_channels = None

  for data_dir in data_dirs:
    # Load the dataset config if it exists (ignore corrupted datasets)
    if os.path.isfile(get_config_filename(data_dir)):
      data_cfg = load_config(data_dir)

      # Backward compatibility
      if not hasattr(data_cfg, 'clean_aux'):
        data_cfg.clean_aux = False

      # Check whether the dataset matches the requirements
      if get_main_feature(data_cfg.features) == get_main_feature(cfg.features) and \
        all(f in data_cfg.features for f in cfg.features) and \
        data_cfg.clean_aux == cfg.clean_aux and \
        data_cfg.transfer == cfg.transfer:
        # Select the most recent version with the minimal amount of channels stored
        num_channels = len(get_dataset_channels(data_cfg.features))
        if best_dir is None or num_channels <= best_num_channels:
          best_dir = data_dir
          best_num_channels = num_channels
  
  if best_dir is None:
    error('no matching preproccessed dataset found')
  return best_dir

class PreprocessedDataset(Dataset):
  def __init__(self, cfg, name):
    super(PreprocessedDataset, self).__init__()

    if not name:
      self.samples = []
      self.num_images = 0
      return

    # Check whether the preprocessed images have all required features
    data_dir = get_preproc_data_dir(cfg, name)
    data_cfg = load_config(data_dir)

    self.tile_size = cfg.tile_size

    # Get the features
    self.features = cfg.features
    self.main_feature = get_main_feature(cfg.features)
    self.aux_features = get_aux_features(cfg.features)
    self.clean_aux = cfg.clean_aux and self.aux_features

    # Get the channels
    self.channels = get_dataset_channels(cfg.features)
    self.all_channels = get_dataset_channels(data_cfg.features)
    self.num_main_channels = len(get_model_channels(self.main_feature))

    # Get the image samples
    samples_filename = os.path.join(data_dir, 'samples.json')
    self.samples = load_json(samples_filename)
    self.num_images = len(self.samples)

    if self.num_images == 0:
      return

    # Create the memory mapping based image reader
    tza_filename = os.path.join(data_dir, 'images.tza')
    self.images = tza.Reader(tza_filename)

## -----------------------------------------------------------------------------
## Training dataset
## -----------------------------------------------------------------------------

class TrainingDataset(PreprocessedDataset):
  def __init__(self, cfg, name):
    super(TrainingDataset, self).__init__(cfg, name)
    self.max_padding = 16

  def __len__(self):
    return self.num_images

  def __getitem__(self, index):
    # Get the input and target images
    input_name, target_name = self.samples[index]
    input_image,  _ = self.images[input_name]
    target_image, _ = self.images[target_name]

    # Get the size of the image
    height = input_image.shape[0]
    width  = input_image.shape[1]
    if height < self.tile_size or width < self.tile_size:
      error('image is smaller than the tile size')

    # Generate a random crop
    sy = sx = self.tile_size
    if rand() < 0.1:
      # Randomly zero pad later to avoid artifacts for images that require padding
      sy -= randint(self.max_padding)
      sx -= randint(self.max_padding)
    oy = randint(height - sy + 1)
    ox = randint(width  - sx + 1)

    # Randomly permute some channels to improve training quality
    input_channels = self.channels[:] # copy

    # Randomly permute the color channels
    color_features = list(set(self.features) & {'hdr', 'ldr', 'alb'})
    if color_features:
      color_order = randperm(3)
      for f in color_features:
        shuffle_channels(input_channels, f+'.r', color_order)

    # Randomly permute the L1 SH coefficients and keep only 3 of them
    if 'sh1' in self.features:
      sh1_order = randperm(9)
      shuffle_channels(input_channels, 'sh1x.r', sh1_order, 3)

    # Randomly permute the normal channels
    if 'nrm' in self.features:
      normal_order = randperm(3)
      shuffle_channels(input_channels, 'nrm.x', normal_order)

    # Get the indices of the input and target channels
    input_channel_indices  = get_channel_indices(input_channels, self.all_channels)
    target_channel_indices = input_channel_indices[:self.num_main_channels]
    #print(input_channels, input_channel_indices)

    # Crop the input and target images
    if self.clean_aux:
      # Get the auxiliary features from the target image
      aux_channel_indices = input_channel_indices[self.num_main_channels:]
      input_image  = input_image [oy:oy+sy, ox:ox+sx, target_channel_indices]
      aux_image    = target_image[oy:oy+sy, ox:ox+sx, aux_channel_indices]
      target_image = target_image[oy:oy+sy, ox:ox+sx, target_channel_indices]
      input_image  = np.concatenate((input_image, aux_image), axis=2)
    else:
      # Get the auxiliary features from the input image
      input_image  = input_image [oy:oy+sy, ox:ox+sx, input_channel_indices]
      target_image = target_image[oy:oy+sy, ox:ox+sx, target_channel_indices]

    # Randomly transform the tiles to improve training quality
    if rand() < 0.5:
      # Flip vertically
      input_image  = np.flip(input_image,  0)
      target_image = np.flip(target_image, 0)

    if rand() < 0.5:
      # Flip horizontally
      input_image  = np.flip(input_image,  1)
      target_image = np.flip(target_image, 1)

    if rand() < 0.5:
      # Transpose
      input_image  = np.swapaxes(input_image,  0, 1)
      target_image = np.swapaxes(target_image, 0, 1)
      sy, sx = sx, sy

    # Zero pad the tiles (always makes a copy)
    pad_size = ((0, self.tile_size - sy), (0, self.tile_size - sx), (0, 0))
    input_image  = np.pad(input_image,  pad_size, mode='constant')
    target_image = np.pad(target_image, pad_size, mode='constant')

    # Randomly zero the main feature channels if there are auxiliary features
    # This prevents "ghosting" artifacts when the main feature is entirely black
    if self.aux_features and rand() < 0.01:
      input_image[:, :, 0:self.num_main_channels] = 0
      target_image[:] = 0

    # DEBUG: Save the tile
    #save_image('tile_%d.png' % index, target_image)

    # Convert the tiles to tensors
    return image_to_tensor(input_image), image_to_tensor(target_image)

## -----------------------------------------------------------------------------
## Validation dataset
## -----------------------------------------------------------------------------

class ValidationDataset(PreprocessedDataset):
  def __init__(self, cfg, name):
    super(ValidationDataset, self).__init__(cfg, name)
    self.tiles = []

    if self.num_images == 0:
      return

    input_channel_indices = get_channel_indices(self.channels, self.all_channels)

    # Split the images into tiles
    for sample_index in range(self.num_images):
      # Get the input image
      input_name,  _ = self.samples[sample_index]
      input_image, _ = self.images[input_name]

      # Get the size of the image
      height = input_image.shape[0]
      width  = input_image.shape[1]
      if height < self.tile_size or width < self.tile_size:
        error('image is smaller than the tile size')

      # Compute the number of tiles
      num_tiles_y = height // self.tile_size
      num_tiles_x = width  // self.tile_size

      # Compute the start offset for centering
      start_y = (height % self.tile_size) // 2
      start_x = (width  % self.tile_size) // 2

      # Add the tiles
      for y in range(num_tiles_y):
        for x in range(num_tiles_x):
          oy = start_y + y * self.tile_size
          ox = start_x + x * self.tile_size

          if self.main_feature == 'sh1':
            for k in range(0, 9, 3):
              ch = input_channel_indices[k:k+3] + input_channel_indices[9:]
              self.tiles.append((sample_index, oy, ox, ch))
          else:
            self.tiles.append((sample_index, oy, ox, input_channel_indices))
      
  def __len__(self):
    return len(self.tiles)

  def __getitem__(self, index):
    # Get the tile
    sample_index, oy, ox, input_channel_indices = self.tiles[index]
    sy = sx = self.tile_size

    # Get the input and target images
    input_name, target_name = self.samples[sample_index]
    input_image,  _ = self.images[input_name]
    target_image, _ = self.images[target_name]

    # Get the indices of target channels
    target_channel_indices = input_channel_indices[:self.num_main_channels]

    # Crop the input and target images
    if self.clean_aux:
      # Get the auxiliary features from the target image
      aux_channel_indices = input_channel_indices[self.num_main_channels:]
      input_image  = input_image [oy:oy+sy, ox:ox+sx, target_channel_indices]
      aux_image    = target_image[oy:oy+sy, ox:ox+sx, aux_channel_indices]
      target_image = target_image[oy:oy+sy, ox:ox+sx, target_channel_indices]
      input_image  = np.concatenate((input_image, aux_image), axis=2)
    else:
      # Get the auxiliary features from the input image
      input_image  = input_image [oy:oy+sy, ox:ox+sx, input_channel_indices]
      target_image = target_image[oy:oy+sy, ox:ox+sx, target_channel_indices]

    # Convert the tiles to tensors
    # Copying is required because PyTorch does not support non-writeable tensors
    return image_to_tensor(input_image.copy()), image_to_tensor(target_image.copy())
