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
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
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


def flow_datagens(datagens, data_points, batch_size, seed=1234):
    iters = [
        datagens[0].flow(
            list_of_images_to_4d_array([dp.x for dp in data_points]),
            # (
            #     list_of_images_to_4d_array([dp.x for dp in data_points]),
            #     np.array([dp.meta for dp in data_points]),
            # ),
            batch_size=batch_size,
            seed=seed,
        ),
        datagens[1].flow(
            list_of_images_to_4d_array([dp.y for dp in data_points]),
            batch_size=batch_size,
            seed=seed,
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


MODEL_CHECKPOINT_NAME = 'model_checkpoint_weights.{epoch:02d}-{val_loss:.2f}.hdf5'


from matplotlib import pyplot as plt


class SaveYsCallback(keras.callbacks.Callback):
    OUTPUT_IMAGE_TEMPLATE = 'training_progress_epoch={}_img={}.jpg'

    def __init__(self, output_path, x_images, y_images):
        self.x_images = x_images
        self.y_images = y_images
        self.output_path = output_path
        utils.mkdir_if_not_exist(self.output_path)

    def on_epoch_end(self, epoch, logs=None):
        for i, (x, y) in enumerate(zip(self.x_images, self.y_images)):
            out_path = os.path.join(self.output_path, self.OUTPUT_IMAGE_TEMPLATE.format(epoch, i))
            print("Saving image {} of {} to {}".format(i, len(self.x_images)-1, out_path))
            y_pred = self.model.predict(x)
            f, ax = plt.subplots(1, 3)

            x = x.squeeze()
            for ci in range(x.shape[2]):
                channel = x[:, :, ci]
                channel -= channel.min()
                x[:, :, ci] = normalize(channel)

            ax[0].imshow(x)
            ax[1].imshow(y_pred[0, :, :, 0])
            ax[2].imshow(y[0, :, :, 0])
            f.savefig(out_path)
            plt.close()


class PrintYsCallback(keras.callbacks.Callback):
    def __init__(self, x_images, y_images):
        self.x_images = x_images
        self.y_images = y_images

    @staticmethod
    def _find_max_location_normed(array):
        col_maxes = array.max(axis=0)
        row_maxes = array.max(axis=1)
        return (row_maxes.argmax(), col_maxes.argmax()), \
               (row_maxes.argmax() / len(row_maxes), col_maxes.argmax() / len(col_maxes))

    def on_epoch_end(self, epoch, logs=None):
        for i, (x, y) in enumerate(zip(self.x_images, self.y_images)):
            print("==========================")
            print(" Starting image {} of {}".format(i, len(self.x_images)))
            print("==========================")
            y_pred = self.model.predict(x)

            y_pred = y_pred.squeeze()
            y = y.squeeze()

            print("y_pred: ")
            print(y_pred)
            print("y_true: ")
            print(y)

            y_pred0 = y_pred[:, :, 0]
            y_pred1 = y_pred[:, :, 1]
            y0 = y[:, :, 0]
            y1 = y[:, :, 1]

            print('y_pred[:,:,0] shape: {}, min: {}, max: {}, mean: {}'.format(
                y_pred0.shape, y_pred0.min(), y_pred0.max(), y_pred0.mean()))
            print('y_pred[:,:,1] shape: {}, min: {}, max: {}, mean: {}'.format(
                y_pred1.shape, y_pred1.min(), y_pred1.max(), y_pred1.mean()))
            print()

            print('y_true[:,:,0] shape: {}, min: {}, max: {}, mean: {}'.format(
                y0.shape, y0.min(), y0.max(), y0.mean()))
            print('y_true[:,:,1] shape: {}, min: {}, max: {}, mean: {}'.format(
                y1.shape, y1.min(), y1.max(), y1.mean()))
            print()

            print('y_pred[:,:,0] max location: row, col = {}, yx = {}'.format(
                *self._find_max_location_normed(y_pred0)))
            print('y_pred[:,:,1] max location: row, col = {}, yx = {}'.format(
                *self._find_max_location_normed(y_pred1)))
            print()

            print('y[:,:,0] max location: row, col = {}, yx = {}'.format(
                *self._find_max_location_normed(y0)))
            print('y[:,:,1] max location: row, col = {}, yx = {}'.format(
                *self._find_max_location_normed(y1)))
            print()
            print()


def get_callbacks(output_dir, x_valid, y_valid):
    patience = 10
    callbacks = [
        keras_addons.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience, min_lr=1e-6),
        keras_addons.ReduceLROnPlateau(monitor='val_loss', factor=1000, patience=patience * 2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, MODEL_CHECKPOINT_NAME),
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
        PrintYsCallback(x_valid, y_valid),
        SaveYsCallback(os.path.join(output_dir, 'progress_images'), x_valid, y_valid),
    ]
    return callbacks


if __name__ == '__main__':
    data_dir = './data/'
    do_resize = False
    do_load_model = False

    df = pd.read_csv(os.path.join(data_dir, 'labels.txt'), delimiter=' ')
    df.columns = ['image', 'x', 'y']

    # y_px, x_px = utils.get_image_size(data_dir)
    # print('y_px={}, x_px={}'.format(y_px, x_px))

    y_px, x_px = (320, 480)
    resized_images_dir = os.path.join(data_dir, 'resized_h={}_w={}'.format(y_px, x_px))
    if do_resize:
        resize_images(data_dir, resized_images_dir, y_px, x_px)

    output_dir = os.path.join(data_dir, '../train_output')
    utils.mkdir_if_not_exist(output_dir)

    # labels_dir = os.path.join(data_dir, 'label_images')
    # utils.mkdir_if_not_exist(labels_dir)

    num_classes = 2
    seed = 1234

    batch_size = 1
    lr = 1e-3
    num_epochs = 100

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
        y = y[:, :, np.newaxis]
        y = np.concatenate([y, 1 - y.copy()], axis=2)
        data_points.append(DataPoint(x=x, y=y, meta=file_name))

    data = train_valid_test_split(data_points, test_prob=0.15, valid_prob=0.15, random_state=seed)

    y_px, x_px = utils.get_image_size(resized_images_dir)
    num_classes = 2
    seed = 1234

    model = model_module.build_compile(optimizer, y_px, x_px, extra_metrics=keras_addons.mode_distance)
    if do_load_model:
        keras_addons.load_model(os.path.join(
            output_dir,
            MODEL_CHECKPOINT_NAME.format(
                epoch=12,
                val_loss=0.06,
            ))
        )

    datagens = make_datagens(data[DataSubsets.train])

    # create images to print in callback
    datagens_flow = {subset: flow_datagens(datagens, data[subset], 1, seed)
                     for subset in DataSubsets}
    x_valid_show = []
    y_valid_show = []
    num_images_to_show = 2
    for i, (x, y) in enumerate(datagens_flow[DataSubsets.valid]):
        if i >= num_images_to_show:
            break
        x_valid_show.append(x)
        y_valid_show.append(y)

    # reset data generators
    datagens_flow = {subset: flow_datagens(datagens, data[subset], batch_size, seed)
                     for subset in DataSubsets}

    steps_per_epoch = len(data[DataSubsets.train]) / batch_size
    history = model.fit_generator(
        datagens_flow[DataSubsets.train],
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        validation_data=datagens_flow[DataSubsets.valid],
        validation_steps=len(data[DataSubsets.valid]) / batch_size,
        callbacks=get_callbacks(output_dir, x_valid_show, y_valid_show),
    )

    print(history)

    for i, (x, y) in enumerate(datagens_flow[DataSubsets.train]):
        print()
        y_pred = model.predict(x)
        print(y_pred.squeeze())
        print(y.shape)
        print(y_pred.shape)
        print(y_pred.min(), y_pred.max(), y_pred.mean())
        if i >= 0:
            break

