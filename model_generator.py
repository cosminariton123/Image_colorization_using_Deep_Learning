import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, ReLU, MaxPool2D, UpSampling2D

from custom_metrics import CUSTOM_METRICS
from util import INPUT_SIZE_NUMPY, GROUND_TRUTH_SIZE_NUMPY

def make_model(model_name):

    input = tf.keras.Input(shape = INPUT_SIZE_NUMPY)

    layer = input

    #All filters and units should be multiple of 8(even better 128 for TPUs) for efficiency
    for _ in range(2):
        layer = Conv2D(16, (3, 3), activation="relu")(layer)

    for _ in range(2):
        layer = Conv2D(32, (3, 3), activation="relu")(layer)

    for _ in range(2):
        layer = Conv2D(128, (3, 3), activation="relu")(layer)

    for _ in range(2):
        layer = Conv2DTranspose(64, (3, 3), activation="relu")(layer)

    for _ in range(2):
        layer = Conv2DTranspose(32, (3, 3), activation="relu")(layer)

    layer = Conv2DTranspose(16, (3, 3), activation="relu")(layer)

    layer = Conv2DTranspose(GROUND_TRUTH_SIZE_NUMPY[2], 3, padding="same", activation="tanh", dtype = tf.float32)(layer)


    optimizer = tf.keras.optimizers.Adam()

    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic = True)

    model = tf.keras.Model(inputs = input, outputs = layer, name = model_name)
    model.compile(loss="mse", optimizer=optimizer, metrics=CUSTOM_METRICS)

    return model
