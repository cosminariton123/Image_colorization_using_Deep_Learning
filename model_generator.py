import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Concatenate

from custom_metrics import CUSTOM_METRICS
from util import INPUT_SIZE_NUMPY, GROUND_TRUTH_SIZE_NUMPY

def make_model():

    input = tf.keras.Input(shape = INPUT_SIZE_NUMPY)

    layer = input

    #All filters and units should be multiple of 8(even better 128 for TPUs) for efficiency
    layer = Conv2D(64, 3, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, 3, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    d1 = layer

    layer = Conv2D(64, 2, strides=2, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    
    layer = Conv2D(128, 3, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, 3, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    d2 = layer

    layer = Conv2D(128, 2, strides=2, activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2D(256, 3, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(256, 3, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    d3 = layer

    layer = Conv2D(256, 2, strides=2, activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2D(512, 3, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(512, 3, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    

    #UP

    layer = Conv2DTranspose(256, 2, strides=2, activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2DTranspose(256, 3, dilation_rate=2, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2DTranspose(256, 3, dilation_rate=2, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Concatenate()([layer, d3])

    layer = Conv2DTranspose(128, 3, strides=2, activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2DTranspose(128, 3, dilation_rate=2, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2DTranspose(128, 3, dilation_rate=2, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Concatenate()([layer, d2])


    layer = Conv2DTranspose(64, 2, strides=2, activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2DTranspose(64, 3, dilation_rate=2, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2DTranspose(64, 3, dilation_rate=2, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Concatenate()([layer, d1])

    layer = Conv2DTranspose(64, 3, dilation_rate=2, activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Concatenate()([layer, input])
    layer = Conv2D(64, 1, activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, 1, activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2D(GROUND_TRUTH_SIZE_NUMPY[2], 1, activation="tanh")(layer)
    layer = BatchNormalization(dtype = tf.float32)(layer)

    layer = (layer + 1) / 2

    layer = layer * 255


    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.00000001,
        momentum=0.85
        #clipvalue=0.2
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic = True)

    model = tf.keras.Model(inputs = input, outputs = layer)
    model.compile(loss="mse", optimizer=optimizer, metrics=CUSTOM_METRICS)

    return model
