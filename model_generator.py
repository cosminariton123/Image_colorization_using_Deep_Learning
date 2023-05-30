import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, ReLU, MaxPool2D, UpSampling2D

from custom_metrics import CUSTOM_METRICS
from util import INPUT_SIZE_NUMPY, GROUND_TRUTH_SIZE_NUMPY

def make_model(model_name):

    input = tf.keras.Input(shape = INPUT_SIZE_NUMPY)

    layer = input

    #All filters and units should be multiple of 8(even better 128 for TPUs) for efficiency
    layer = Conv2D(128, 3)(layer)
    layer = ReLU()(layer)
    layer = Conv2D(128, 3)(layer)
    layer = ReLU()(layer)
    d1 = layer

    
    layer = MaxPool2D()(layer)

    layer = Conv2D(256, 3)(layer)
    layer = ReLU()(layer)
    layer = Conv2D(256, 3)(layer)
    layer = ReLU()(layer)
    d2 = layer

    layer = MaxPool2D()(layer)

    layer = Conv2D(512, 3)(layer)
    layer = ReLU()(layer)
    layer = Conv2D(512, 3)(layer)
    layer = ReLU()(layer)
    

    #UP
    layer = UpSampling2D()(layer)

    layer = Conv2DTranspose(256, 3, dilation_rate=2)(layer)
    layer = ReLU()(layer)
    layer = Conv2DTranspose(256, 3, dilation_rate=2)(layer)
    layer = ReLU()(layer)
    layer = Conv2DTranspose(256, 2)(layer)
    layer = ReLU()(layer)

    layer += d2

    layer = UpSampling2D()(layer)

    layer = Conv2DTranspose(128, 3, dilation_rate=2)(layer)
    layer = ReLU()(layer)
    layer = Conv2DTranspose(128, 3, dilation_rate=2)(layer)
    layer = ReLU()(layer)

    layer += d1


    layer = Conv2DTranspose(128, 3)(layer)
    layer = ReLU()(layer)
    layer = Conv2DTranspose(128, 3)(layer)
    layer = ReLU()(layer)

    layer =  Concatenate()([layer, input])

    layer = Conv2D(128, 3, padding="same")(layer)
    layer = ReLU()(layer)
    layer = Conv2D(128, 3, padding="same")(layer)
    layer = ReLU()(layer)

    layer = Conv2D(GROUND_TRUTH_SIZE_NUMPY[2], 3, padding="same", activation="tanh", dtype = tf.float32)(layer)


    #if GROUND_TRUTH_SIZE_NUMPY[2] == 2:
    #    input = tf.cast(input, dtype=tf.float32)
    #    input = input / 255
    #    layer = layer / 2
    #    layer = Concatenate()([input, layer])
    #    layer = tf.image.yuv_to_rgb(layer)
    #    layer = layer * 255

    #else:
    #    layer = (layer + 1) / 2 * 255


    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.0000001,
        momentum=0.99,
        clipvalue = 2,
        decay = 0.001
    )
    #optimizer = tf.keras.optimizers.Adam()

    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic = True)

    #layer = Conv2D(GROUND_TRUTH_SIZE_NUMPY[2], 3, activation="tanh", dtype=tf.float32, padding="same")(input)

    model = tf.keras.Model(inputs = input, outputs = layer, name = model_name)
    model.compile(loss="mse", optimizer=optimizer, metrics=CUSTOM_METRICS)

    return model
