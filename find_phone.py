from __future__ import division
import argparse
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)

from skimage import io

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import keras_addons
import train_phone_finder
import utils

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "img_file",
        help="path to the image file within which the phone should be found",
    )
    args = ap.parse_args()

    datagens = utils.load_pickle(train_phone_finder.DATAGENS_FILENAME)
    x_raw = io.imread(args.img_file)
    x = keras_addons.preproc_x(x_raw, datagens[0])

    model_template = os.path.join(train_phone_finder.TRAIN_OUTPUT_DIR, train_phone_finder.MODEL_CHECKPOINT_NAME)
    model = keras_addons.load_model(model_template.format(epoch=3))

    y_pred = model.predict(x, batch_size=1).squeeze()[:, :, 0]
    row, col = keras_addons.find_normed_maxes(y_pred)
    print("{:0.4f} {:0.4f}".format(col, row))
