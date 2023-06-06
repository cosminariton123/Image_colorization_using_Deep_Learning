import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, ReLU, LeakyReLU, ELU, MaxPool2D, UpSampling2D, Dropout, BatchNormalization

from custom_metrics import CUSTOM_METRICS
from util import INPUT_SIZE_NUMPY, GROUND_TRUTH_SIZE_NUMPY

def make_model(model_name):

    input = tf.keras.Input(shape = INPUT_SIZE_NUMPY)

    layer = input

    #All filters and units should be multiple of 8(even better 128 for TPUs) for efficiency
    layer = Conv2D(64, 3, activation="relu", padding="same")(layer)
    d1 = Conv2D(64, 3, activation="relu", padding="same")(layer)

    layer = MaxPool2D()(d1)

    layer = Conv2D(128, 3, activation="relu", padding="same")(layer)
    d2 = Conv2D(128, 3, activation="relu", padding="same")(layer)

    layer = MaxPool2D()(d2)

    layer = Conv2D(256, 3, activation="relu", padding="same")(layer)
    d3 = Conv2D(256, 3, activation="relu", padding="same")(layer)

    layer = MaxPool2D()(d3)

    layer = Conv2D(512, 3, activation="relu", padding="same")(layer)
    d4 = Conv2D(512, 3, activation="relu", padding="same")(layer)

    layer = MaxPool2D()(d4)

    layer = Conv2D(1024, 3, activation="relu", padding="same")(layer)
    layer = Conv2D(1024, 3, activation="relu", padding="same")(layer)

    layer = Conv2DTranspose(512, 2, strides=2, activation="relu")(layer)

    layer = Concatenate()([layer, d4])
    layer = Conv2D(512, 1, activation="relu")(layer)

    layer = Conv2D(512, 3, activation="relu", padding="same")(layer)
    layer = Conv2D(512, 3, activation="relu", padding="same")(layer)

    layer = Conv2DTranspose(256, 3, strides=2, activation="relu")(layer)
    

    layer = Concatenate()([layer, d3])
    layer = Conv2D(256, 1, activation="relu")(layer)

    layer = Conv2D(256, 3, activation="relu", padding="same")(layer)
    layer = Conv2D(256, 3, activation="relu", padding="same")(layer)

    
    layer = Conv2DTranspose(128, 3, strides=2, activation="relu")(layer)

    
    layer = Concatenate()([layer, d2])
    layer = Conv2D(128, 1, activation="relu")(layer)

    layer = Conv2D(128, 3, activation="relu", padding="same")(layer)
    layer = Conv2D(128, 3, activation="relu", padding="same")(layer)


    layer = Conv2DTranspose(64, 2, strides=2, activation="relu")(layer)

    
    layer = Concatenate()([layer, d1])
    layer = Conv2D(64, 1, activation="relu")(layer)

    layer = Conv2D(64, 3, activation="relu", padding="same")(layer)
    layer = Conv2D(64, 3, activation="relu", padding="same")(layer)


    layer = Conv2D(GROUND_TRUTH_SIZE_NUMPY[2], 3, activation="tanh", dtype = tf.float32, padding="same")(layer)

    optimizer = tf.optimizers.Adam()

    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic = True)



    model = tf.keras.Model(inputs = input, outputs = layer, name = model_name)
    model.compile(loss="mse", optimizer=optimizer, metrics=CUSTOM_METRICS)

    return model
    