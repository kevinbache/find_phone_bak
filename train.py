from __future__ import division

from tensorflow.python import keras

import os
import glob
from enum import Enum
from collections import namedtuple

from PIL import Image

import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal
from skimage import io
from sklearn.model_selection import train_test_split

# from tensorflow.python import keras
# import keras
# from tensorflow.python.keras import callbacks
# from keras import callbacks
import keras_addons
import model as model_module
import utils

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(current_dir)
for p in sys.path:
    print(p)

class DataSubsets(Enum):
    """Subsets of a data set"""
    train = 1
    valid = 2
    test = 3


DataPoint = namedtuple("DataPoint", ['x', 'y', 'meta'])


def resize_images(input_dir, output_dir, new_height, new_width):
    files = glob.glob(os.path.join(input_dir, '*.jpg'))
    utils.mkdir_if_not_exist(output_dir)
    for fi, full_file in enumerate(files):
        print('Resizing image {} of {}'.format(fi + 1, len(files)))
        im = Image.open(full_file)
        im_resized = im.resize((new_width, new_height), Image.ANTIALIAS)
        out_full_file = os.path.join(output_dir, os.path.basename(full_file))
        im_resized.save(out_full_file, 'JPEG', quality=90)


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


def read_x_data(data_dir):
    files = glob.glob(os.path.join(data_dir, '*.jpg'))
    return [(os.path.basename(file), io.imread(file)) for file in files]


def list_of_images_to_4d_array(list_of_image_arrays):
    ndim = list_of_image_arrays[0].ndim
    if ndim == 3:
        return np.concatenate([im[np.newaxis, :, :, :] for im in list_of_image_arrays], axis=0)
    elif ndim == 2:
        return np.concatenate([im[np.newaxis, :, :, np.newaxis] for im in list_of_image_arrays], axis=0)
    else:
        raise ValueError('got ndim = {}, only set up to handle 2 and 3'.format(ndim))


def make_datagens(data_points):
    data_gen_args = dict(
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
    )

    x_datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        **data_gen_args,
    )

    y_datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        **data_gen_args,
    )

    x_datagen.fit(list_of_images_to_4d_array([dp.x for dp in data_points]))
    y_datagen.fit(list_of_images_to_4d_array([dp.y for dp in data_points]))

    return [x_datagen, y_datagen]


def flow_datagens(datagens, data_points, batch_size):
    iters = [
        datagens[0].flow(
            list_of_images_to_4d_array([dp.x for dp in data_points]),
            # (
            #     list_of_images_to_4d_array([dp.x for dp in data_points]),
            #     np.array([dp.meta for dp in data_points]),
            # ),
            batch_size=batch_size,
        ),
        datagens[1].flow(
            list_of_images_to_4d_array([dp.y for dp in data_points]),
            batch_size=batch_size,
        )
    ]
    return zip(*iters)


def train_valid_test_split(data_points, test_prob, valid_prob, random_state=1234):
    test_prob_full = test_prob / (1 - valid_prob)
    train, valid = train_test_split(data_points, test_size=valid_prob, random_state=random_state)
    train, test = train_test_split(train, test_size=test_prob_full, random_state=random_state)
    return {
        DataSubsets.train: train,
        DataSubsets.valid: valid,
        DataSubsets.test: test,
    }


def get_callbacks(output_dir):
    patience = 10
    callbacks = [
        keras_addons.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience, min_lr=1e-6),
        keras_addons.ReduceLROnPlateau(monitor='val_loss', factor=1000, patience=patience * 2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'model_checkoint_weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
            monitor='val_loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=1),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=30,
            verbose=1,
            mode='auto',
            # baseline=None,
            # restore_best_weights=False
        ),
    ]
    return callbacks


if __name__ == '__main__':
    data_dir = './data/'

    df = pd.read_csv(os.path.join(data_dir, 'labels.txt'), delimiter=' ')
    df.columns = ['image', 'x', 'y']

    # y_px, x_px = utils.get_image_size(data_dir)
    # print('y_px={}, x_px={}'.format(y_px, x_px))

    y_px, x_px = (320, 480)
    resized_images_dir = os.path.join(data_dir, 'resized_h={}_w={}'.format(y_px, x_px))
    resize_images(data_dir, resized_images_dir, y_px, x_px)

    output_dir = os.path.join(data_dir, 'training_output')

    # labels_dir = os.path.join(data_dir, 'label_images')
    # utils.mkdir_if_not_exist(labels_dir)

    num_classes = 2
    seed = 1234

    batch_size = 1
    lr = 0.001
    num_epochs = 30

    sigma = 0.0005

    # optimizer = keras.optimizers.Adam(lr=lr)
    optimizer = 'adam'

    x_info = read_x_data(resized_images_dir)

    # create y data
    data_points = []
    for file_name, x in x_info:
        row = df[df['image'] == file_name]
        if len(row) == 0:
            continue
        y = make_gaussian_label_image(row['x'].values[0], row['y'].values[0], x_px, y_px, sigma)
        y = normalize(y)
        y = np.concatenate([y[:, :, np.newaxis], 1 - y.copy()[:, :, np.newaxis]], axis=2)
        data_points.append(DataPoint(x=x, y=y, meta=file_name))

    data = train_valid_test_split(data_points, test_prob=0.15, valid_prob=0.15, random_state=seed)

    y_px, x_px = utils.get_image_size(resized_images_dir)
    num_classes = 2
    seed = 1234

    model = model_module.build_compile(optimizer, y_px, x_px)

    datagens = make_datagens(data[DataSubsets.train])
    datagens_flow = {subset: flow_datagens(datagens, data[subset], batch_size)
                     for subset in DataSubsets}


    steps_per_epoch = len(data[DataSubsets.train]) / batch_size
    # steps_per_epoch = 2
    history = model.fit_generator(
        datagens_flow[DataSubsets.train],
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        validation_data=datagens_flow[DataSubsets.valid],
        validation_steps=len(data[DataSubsets.valid]) / batch_size,
        callbacks = get_callbacks(output_dir),
    )

    print(history)

    for i, (x, y) in enumerate(datagens_flow[DataSubsets.train]):
        print()
        y_pred = model.predict(x)
        print(y_pred.squeeze())
        print(y.shape)
        print(y_pred.shape)
        print(y_pred.min(), y_pred.max(), y_pred.mean())
        if i > 5:
            break

