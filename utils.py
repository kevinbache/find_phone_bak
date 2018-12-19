import os
import pickle
from typing import Text

from skimage import io


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
