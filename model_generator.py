import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization, Dropout, ReLU, Concatenate, Activation

from custom_metrics import CUSTOM_METRICS
from util import INPUT_SIZE_NUMPY, GROUND_TRUTH_SIZE_NUMPY

def make_model():

    input = tf.keras.Input(shape = INPUT_SIZE_NUMPY)

    layer = input

    layer = Conv2D(128, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, 3, padding="same", activation="relu")(layer)
    d1 = layer

    layer = Conv2D(128, 3, strides=2, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2D(256, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(256, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    d2 = layer

    layer = Conv2D(256, 3, 2, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2D(512, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(512, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(512, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(512, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2DTranspose(256, 2, 2, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Concatenate()([d2, layer])

    layer = Conv2D(256, 1, activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2D(256, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(256, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(256, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2DTranspose(128, 2, 2, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Concatenate()([d1, layer])

    layer= Conv2D(128, 1, activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2D(128, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Concatenate()([input, layer])
    layer = Conv2D(128, 1, activation="relu")(layer)

    layer = Conv2D(128, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, 3, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2D(GROUND_TRUTH_SIZE_NUMPY[2], 3, padding="same", activation="tanh")(layer)

    layer = (layer + 1) / 2

    layer = layer * 255



    model = tf.keras.Model(inputs = input, outputs = layer)
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=CUSTOM_METRICS)

    return model
