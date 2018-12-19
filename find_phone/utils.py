import os
import pickle
from collections import namedtuple
from enum import Enum
from typing import Text

import numpy as np
from skimage import io


class DataSubsets(Enum):
    """Subsets of a data set"""
    train = 1
    valid = 2
    test = 3


DataPoint = namedtuple("DataPoint", ['x', 'y', 'meta'])


def mkdir_if_not_exist(directory_path):
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def append_before_ext(fullfile: Text, thing_to_append: Text):
    """append something to a filename right before its extension"""
    base, ext = os.path.splitext(fullfile)
    return '{}{}{}'.format(base, thing_to_append, ext)


def save_pickle(obj_to_save: object, save_location: Text):
    with open(save_location, 'wb') as f:
        pickle.dump(obj_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()


def load_pickle(load_location: Text):
    with open(load_location, 'rb') as f:
        out = pickle.load(f)
        f.close()
    return out


def get_image_size(data_dir):
    img = io.imread(os.path.join(data_dir, '0.jpg'))
    y_px, x_px, _ = img.shape
    return y_px, x_px


def normalize(a, new_max=1.0):
    """normalize the values in the input array to the range [0, new_max]"""
    a = (a - a.min())
    a = a/a.max()
    a *= new_max
    return a


def list_of_images_to_4d_array(list_of_image_arrays):
    ndim = list_of_image_arrays[0].ndim
    if ndim == 3:
        return np.concatenate([im[np.newaxis, :, :, :] for im in list_of_image_arrays], axis=0)
    elif ndim == 2:
        return np.concatenate([im[np.newaxis, :, :, np.newaxis] for im in list_of_image_arrays], axis=0)
    else:
        raise ValueError('got ndim = {}, only set up to handle 2 and 3'.format(ndim))