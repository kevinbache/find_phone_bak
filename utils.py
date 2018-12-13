import os
from typing import Text

from skimage import io


def mkdir_if_not_exist(directory_path):
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def append_before_ext(fullfile: Text, thing_to_append: Text):
    """append something to a filename right before its extension"""
    base, ext = os.path.splitext(fullfile)
    return '{}{}{}'.format(base, thing_to_append, ext)


def get_image_size(data_dir):
    img = io.imread(os.path.join(data_dir, '0.jpg'))
    y_px, x_px, _ = img.shape
    return y_px, x_px

