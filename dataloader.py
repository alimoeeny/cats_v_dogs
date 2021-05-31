import fnmatch
import os
import math

import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
import numpy as np


class CNDLoader(tf.keras.utils.Sequence):
  def __init__(self, image_side, batch_size, root_dir):
    self.image_side = image_side
    self.root_dir = root_dir
    self.batch_size = batch_size
    self.x = []
    self.y = []

    for root, _, filenames in os.walk(root_dir):
      for filename in fnmatch.filter(filenames, '*.jpg'):
        self.x.append(os.path.join(root, filename))
        if 'cat' in filename:
          self.y.append(0)
        elif 'dog' in filename:
          self.y.append(1)
        else:
          raise Exception(f"WTF is this {filename}")

    self.shuffled_indexes = np.random.permutation(len(self.x))

  def __len__(self):
    return math.ceil(len(self.x) / self.batch_size)

  def __getitem__(self, idx):
    if idx == -1: # give me ALL THE THINGS
      indexes = range(0, len(self.x))
    else:
      if idx >= self.__len__():
        idx = idx % self.__len__()
      indexes = range(idx * self.batch_size,(idx + 1) * self.batch_size)

    batch_x = [self.x[self.shuffled_indexes[i]] for i in indexes]
    batch_y = [self.y[self.shuffled_indexes[i]] for i in indexes]

    raw_images = [imread(file_name) for file_name in batch_x]

    scaled_images = [resize(img, (self.image_side, self.image_side), mode='constant') for img in raw_images]

    return np.array(scaled_images), np.array(batch_y), batch_x