from __future__ import division
import os

import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal
from skimage import io

from utils import mkdir_if_not_exist, append_before_ext, get_image_size


def make_gaussian_label_image(mu_x, mu_y, num_x_px, num_y_px, var=0.05):
    """make a 1-d label image of shape (num_y_px, num_x_px) with a gaussian of size defined by var """
    x = np.linspace(0, 1, num_x_px)
    y = np.linspace(0, 1, num_y_px)
    xs, ys = np.meshgrid(x, y)
    locs = np.concatenate([xs[:, :, np.newaxis], ys[:, :, np.newaxis]], axis=2)
    locs_reshaped = locs.reshape(np.prod(locs.shape[:-1]), locs.shape[-1])

    rv = multivariate_normal([mu_x, mu_y], [[var, 0], [0, var]])
    return rv.pdf(locs_reshaped).reshape(locs.shape[:-1])


def normalize(a, new_max=1.0):
    """normalize the values in the input array to the range [0, new_max]"""
    a = (a - a.min())
    a = a/a.max()
    a *= new_max
    return a


if __name__ == '__main__':
    data_dir = './data/'

    df = pd.read_csv(os.path.join(data_dir, 'labels.txt'), delimiter=' ')
    df.columns = ['image', 'x', 'y']

    y_px, x_px = get_image_size(data_dir)

    labels_dir = os.path.join(data_dir, 'label_images')
    mkdir_if_not_exist(labels_dir)

    sigma = 0.0005

    for row_id, row in df.iterrows():
        zs = make_gaussian_label_image(row['x'], row['y'], x_px, y_px, sigma)
        zs = normalize(zs)
        label_filename = append_before_ext(row['image'], '_label')
        io.imsave(os.path.join(labels_dir, label_filename), zs)
