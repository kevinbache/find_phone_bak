from __future__ import division
import os
import warnings

import numpy as np
from matplotlib import pyplot as plt

import keras
from keras import backend as K
from keras import callbacks

from deeplab.model import relu6, BilinearUpsampling
import utils


########
# MISC #
########
def load_model(model_path):
    return keras.models.load_model(
        model_path,
        custom_objects={
            'relu6': relu6,
            'BilinearUpsampling': BilinearUpsampling
        }
    )


###########
# METRICS #
###########
def keras_find_normed_maxes(tensor):
    col_maxes = K.max(tensor, axis=0)
    row_maxes = K.max(tensor, axis=1)

    normed_col_max = K.cast(K.argmax(col_maxes), 'float32') / K.cast(K.shape(col_maxes)[0], 'float32')
    normed_row_max = K.cast(K.argmax(row_maxes), 'float32') / K.cast(K.shape(row_maxes)[0], 'float32')

    return normed_row_max, normed_col_max


def keras_distance(p1, p2):
    return K.sqrt(K.pow(p1[0] - p2[0], 2) + K.pow(p1[1] - p2[1], 2))


def mode_distance(y_true, y_pred):
    return keras_distance(
        keras_find_normed_maxes(y_true[:, :, 0]),
        keras_find_normed_maxes(y_pred[:, :, 0])
    )


def eval_dist(dp: utils.DataPoint, model: keras.models.Model, datagen: keras.preprocessing.image.ImageDataGenerator):
    """returns the distance between predicted location and true location"""
    x = datagen.standardize(dp.x.astype('float32'))[np.newaxis, :, :, :]
    y_pred = model.predict(x, batch_size=1).squeeze()
    return K.eval(mode_distance(dp.y, y_pred))


#############
# CALLBACKS #
#############
class ReduceLROnPlateau(callbacks.Callback):
    """
    KBNOTE: THIS IS A VERSION OF ReduceLROnPlateau which has the factor check disabled.

    Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    # Example

    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```

    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        min_delta: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0,
                 **kwargs):
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        # KBNOTE: I copied this to turn off these lines.
        # if factor >= 1.0:
        #     raise ValueError('ReduceLROnPlateau '
        #                      'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            warnings.warn('`epsilon` argument is deprecated and '
                          'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                  'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


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
                x[:, :, ci] = utils.normalize(channel)

            ax[0].imshow(x)
            ax[1].imshow(y_pred[0, :, :, 0])
            ax[2].imshow(y[0, :, :, 0])
            f.savefig(out_path)
            plt.close()


class PrintYsCallback(keras.callbacks.Callback):
    """Debug helper callback which prints stats on y variables"""
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

            y_pred0 = y_pred[:, :, 0]
            y_pred1 = y_pred[:, :, 1]
            y0 = y[:, :, 0]
            y1 = y[:, :, 1]

            print('y_pred[:,:,1] shape: {}, min: {}, max: {}, mean: {}'.format(
                y_pred1.shape, y_pred1.min(), y_pred1.max(), y_pred1.mean()))
            print('y_true[:,:,1] shape: {}, min: {}, max: {}, mean: {}'.format(
                y1.shape, y1.min(), y1.max(), y1.mean()))

            print('y_pred[:,:,1] max location: row, col = {}, yx = {}'.format(
                *self._find_max_location_normed(y_pred1)))
            print('y[:,:,1]      max location: row, col = {}, yx = {}'.format(
                *self._find_max_location_normed(y1)))
            print()

            print('y_pred[:,:,0] shape: {}, min: {}, max: {}, mean: {}'.format(
                y_pred0.shape, y_pred0.min(), y_pred0.max(), y_pred0.mean()))
            print('y_true[:,:,0] shape: {}, min: {}, max: {}, mean: {}'.format(
                y0.shape, y0.min(), y0.max(), y0.mean()))

            print('y_pred[:,:,0] max location: row, col = {}, yx = {}'.format(
                *self._find_max_location_normed(y_pred0)))
            print('y[:,:,0]      max location: row, col = {}, yx = {}'.format(
                *self._find_max_location_normed(y0)))
            print()
            print()

