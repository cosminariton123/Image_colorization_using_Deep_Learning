import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose

from custom_metrics import CUSTOM_METRICS
from util import INPUT_SIZE_NUMPY, GROUND_TRUTH_SIZE_NUMPY

def make_model():
    input = tf.keras.Input(shape = INPUT_SIZE_NUMPY)

    layer = input


    d1 = Conv2D(64, (3, 3), activation="relu")(layer)
    d2 = Conv2D(64, (3, 3), activation="relu")(d1)
    

    d3 = Conv2D(64, (3, 3), strides=2, activation="relu")(d2)
    d4 = Conv2D(64, (3, 3), activation="relu")(d3)

    d5 = Conv2D(128, (3, 3), strides=2, activation="relu")(d4)
    
    d6 = Conv2D(256, (3, 3), padding="same",activation="relu")(d5)


    s1 = Conv2D(350, (3, 3), padding="same",activation="relu")(d6)
    layer = Conv2D(350, (3, 3), padding="same",activation="relu")(s1)
    layer = layer + s1

    layer = Conv2DTranspose(256, (3, 3), activation="relu")(layer)
    layer = Conv2D(256, (3, 3), activation="relu")(layer)
    layer = layer + d6

    layer = Conv2DTranspose(128, (3, 3), activation="relu")(layer)
    layer = Conv2D(128, (3, 3), activation="relu")(layer)
    layer = layer + d5

    layer = Conv2DTranspose(128, (3, 3), strides=2, activation="relu")(layer)
    layer = Conv2DTranspose(64, (2, 2), activation="relu")(layer)
    layer = layer + d4
    
    layer = Conv2DTranspose(64, (3, 3), activation="relu")(layer)
    layer = layer + d3

    layer = Conv2DTranspose(64, (4, 4), strides=2, activation="relu")(layer)
    layer = layer + d2

    layer = Conv2DTranspose(64, (3, 3), activation="relu")(layer)
    layer = layer + d1
    
    layer = Conv2DTranspose(3, (3, 3), activation="tanh")(layer)

    layer = layer + input

    layer = tf.maximum(layer, 0)

    layer = tf.minimum(layer, 1)

    layer = layer * 255


                
    model = tf.keras.Model(inputs = input, outputs = layer)
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=CUSTOM_METRICS)

    return model
