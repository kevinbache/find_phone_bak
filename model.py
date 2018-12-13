import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np

from tensorflow.python import keras
# import keras

# from keras.models import *
# from keras.layers import *
# from keras.optimizers import *
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as keras


# def unet(pretrained_weights=None, input_size=(256, 256, 1)):
def build_compile(optimizer, input_height=360, input_width=480):
    inputs = keras.layers.Input((input_height, input_width, 3))
    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = keras.layers.Dropout(0.5)(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = keras.layers.Dropout(0.5)(conv5)

    up6 = keras.layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        keras.layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = keras.layers.concatenate([drop4, up6], axis=3)
    conv6 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        keras.layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        keras.layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        keras.layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = keras.models.Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # # model.summary()
    #
    # if (pretrained_weights):
    #     model.load_weights(pretrained_weights)

    return model


# from keras.models import Sequential
# from keras.layers import Reshape
# from keras.layers import Merge
# from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
# from keras.layers.normalization import BatchNormalization
# from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
# from keras.layers.convolutional import Convolution2D, keras.layers.MaxPooling2D, UpSampling2D, ZeroPadding2D
# from keras.layers.convolutional import Convolution1D, MaxPooling1D
# from keras.layers.recurrent import LSTM
# from keras.layers.advanced_activations import LeakyReLU
# from keras.optimizers import Adam , SGD
# from keras.layers.embeddings import Embedding
# from keras.utils import np_utils
# # from keras.regularizers import ActivityRegularizer
# from keras import backend as K


# def segnet(nClasses, optimizer=None, input_height=360, input_width=480):
#     #
#     # kernel = 3
#     # filter_size = 64
#     # pad = 1
#     # pool_size = 2
#     #
#     # model = models.Sequential()
#     # model.add(Layer(input_shape=(3, input_height , input_width )))
#     #
#     # # encoder
#     # model.add(ZeroPadding2D(padding=(pad ,pad)))
#     # model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
#     # model.add(BatchNormalization())
#     # model.add(Activation('relu'))
#     # model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
#     #
#     # model.add(ZeroPadding2D(padding=(pad ,pad)))
#     # model.add(Convolution2D(128, kernel, kernel, border_mode='valid'))
#     # model.add(BatchNormalization())
#     # model.add(Activation('relu'))
#     # model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
#     #
#     # model.add(ZeroPadding2D(padding=(pad ,pad)))
#     # model.add(Convolution2D(256, kernel, kernel, border_mode='valid'))
#     # model.add(BatchNormalization())
#     # model.add(Activation('relu'))
#     # model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
#     #
#     # model.add(ZeroPadding2D(padding=(pad ,pad)))
#     # model.add(Convolution2D(512, kernel, kernel, border_mode='valid'))
#     # model.add(BatchNormalization())
#     # model.add(Activation('relu'))
#     #
#     #
#     # # decoder
#     # model.add( ZeroPadding2D(padding=(pad ,pad)))
#     # model.add( Convolution2D(512, kernel, kernel, border_mode='valid'))
#     # model.add( BatchNormalization())
#     #
#     # model.add( UpSampling2D(size=(pool_size ,pool_size)))
#     # model.add( ZeroPadding2D(padding=(pad ,pad)))
#     # model.add( Convolution2D(256, kernel, kernel, border_mode='valid'))
#     # model.add( BatchNormalization())
#     #
#     # model.add( UpSampling2D(size=(pool_size ,pool_size)))
#     # model.add( ZeroPadding2D(padding=(pad ,pad)))
#     # model.add( Convolution2D(128, kernel, kernel, border_mode='valid'))
#     # model.add( BatchNormalization())
#     #
#     # model.add( UpSampling2D(size=(pool_size ,pool_size)))
#     # model.add( ZeroPadding2D(padding=(pad ,pad)))
#     # model.add( Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
#     # model.add( BatchNormalization())
#     #
#     # model.add(Convolution2D( nClasses , 1, 1, border_mode='valid' ,))
#     #
#     # model.outputHeight = model.output_shape[-2]
#     # model.outputWidth = model.output_shape[-1]
#     #
#     # model.add(Reshape(( nClasses ,  model.output_shape[-2 ] *model.output_shape[-1]   ), input_shape=( nClasses , model.output_shape[-2], model.output_shape[-1]  )))
#     #
#     # model.add(Permute((2, 1)))
#     # model.add(Activation('softmax'))
#     # # model.add(Activation('sigmoid'))
#     #
#     # if not optimizer is None:
#     #     model.compile(loss="categorical_crossentropy", optimizer= optimizer , metrics=['accuracy'] )
#     model = None
#     return model
#
#
# def build_compile(optimizer, input_height=360, input_width=480):
#     pass
