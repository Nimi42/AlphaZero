from keras import losses
from keras.layers import Input, Add, Dense, Conv2D, Activation, BatchNormalization, Flatten, \
    MaxPooling2D, Reshape, Concatenate
from keras.models import Model

from keras.initializers import glorot_uniform
import tensorflow as tf


def chess_conv():
    inp_p1 = Input(shape=(8, 8, 6))
    inp_p2 = Input(shape=(8, 8, 6))
    scalars = Input(shape=(8,))

    with tf.device('/device:CPU:0'):
        # Stage 1
        X = Conv2D(64, (3, 3), name='conv1a', kernel_initializer=glorot_uniform(seed=0))(inp_p1)
        X = BatchNormalization(axis=3, name='bn_conv1a')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2))(X)
        X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='alpha', s=1)
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='beta')
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='gamma')
    with tf.device('/device:GPU:0'):
        Y = Conv2D(64, (3, 3), name='conv1b', kernel_initializer=glorot_uniform(seed=0))(inp_p2)
        Y = BatchNormalization(axis=3, name='bn_conv1b')(Y)
        Y = Activation('relu')(Y)
        Y = MaxPooling2D((2, 2))(Y)
        Y = convolutional_block(Y, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        Y = identity_block(Y, 3, [64, 64, 256], stage=2, block='b')
        Y = identity_block(Y, 3, [64, 64, 256], stage=2, block='c')

    out = Flatten()(Add()([X, Y]))
    out = Concatenate()([out, scalars])

    out = Dense(8*8*8*8, activation='softmax')(out)
    out = Reshape((8, 8, 8, 8))(out)

    model = Model(inputs=[inp_p1, inp_p2, scalars], outputs=out)
    model.compile(loss=losses.mean_squared_error, optimizer="adam")
    model.summary()
    return model


def identity_block(X, f, filters, stage, block):
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X
