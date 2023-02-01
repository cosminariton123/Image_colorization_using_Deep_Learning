import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Dropout, ReLU, Concatenate, Activation

from custom_metrics import CUSTOM_METRICS
from util import INPUT_SIZE_NUMPY, GROUND_TRUTH_SIZE_NUMPY

def make_model():

    input = tf.keras.Input(shape = INPUT_SIZE_NUMPY)

    layer = input


    layer = Conv2D(64, 3)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer = ReLU()(layer)

    layer = Conv2D(64, 3)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer = ReLU()(layer)
    d1 = layer



    layer = Conv2D(64, 3, strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer = ReLU()(layer)
    


    layer = Conv2D(128, 3)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)
    layer = ReLU()(layer)

    layer = Conv2D(128, 3)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)
    layer = ReLU()(layer)
    d2 = layer


    layer = Conv2D(128, 3, strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer =  Dropout(0.3)(layer)
    layer = ReLU()(layer)
    
    layer = Conv2D(256, 3)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.4)(layer)
    layer = ReLU()(layer)

    layer = Conv2D(256, 3)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.4)(layer)
    layer = ReLU()(layer)


    

    layer = Conv2DTranspose(128, 4, strides=3)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)
    layer = ReLU()(layer)
    layer = Concatenate()([layer, d2])

    layer = Conv2D(128, 4)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)
    layer = ReLU()(layer)
    
    layer = Conv2D(128, 3)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)
    layer = ReLU()(layer)


    layer = Conv2DTranspose(64, 3, strides=3)(layer)
    layer = BatchNormalization()(layer)
    layer =  Dropout(0.2)(layer)
    layer = ReLU()(layer)
    layer = Concatenate()([layer, d1])

    layer = Conv2D(128, 3)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer = ReLU()(layer)

    layer = Conv2D(128, 3)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer = ReLU()(layer)

    
    layer = Conv2DTranspose(64, 3, dilation_rate=4)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer= ReLU()(layer)
    layer = Concatenate()([layer, input])


    layer = Conv2D(GROUND_TRUTH_SIZE_NUMPY[2], 1)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation(activation="tanh")(layer)

    layer = (layer + 1) / 2

    layer = layer * 255



    model = tf.keras.Model(inputs = input, outputs = layer)
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=CUSTOM_METRICS)

    return model
