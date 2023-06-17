import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, ReLU, LeakyReLU,  ELU, MaxPool2D, UpSampling2D, Dropout, BatchNormalization

from custom_metrics import CUSTOM_METRICS
from util import INPUT_SIZE_NUMPY, GROUND_TRUTH_SIZE_NUMPY



def make_model(model_name):

    input = tf.keras.Input(shape = (None, None, INPUT_SIZE_NUMPY[-1]))

    layer = input

    #All filters and units should be multiple of 8(even better 128 for TPUs) for efficiency
    layer = Conv2D(64, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(64, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    d1 = ReLU()(layer)

    layer = MaxPool2D()(d1)

    layer = Conv2D(128, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(128, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    d2 = ReLU()(layer)

    layer = MaxPool2D()(d2)
   

    layer = Conv2D(256, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(256, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    d3 = ReLU()(layer)


    layer = MaxPool2D()(d3)

    layer = Conv2D(512, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(512, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    d4 = ReLU()(layer)

    
    layer = MaxPool2D()(d4)

    layer = Conv2D(1024, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(1024, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)



    layer = Conv2DTranspose(256, 2, strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    
    layer = Concatenate()([layer, d4])
    layer = Conv2D(512, 1)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = Conv2D(512, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(512, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)



    layer = Conv2DTranspose(256, 3, strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    
    layer = Concatenate()([layer, d3])
    layer = Conv2D(256, 1)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = Conv2D(256, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(256, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)


    layer = Conv2DTranspose(128, 3, strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    
    layer = Concatenate()([layer, d2])
    layer = Conv2D(128, 1)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = Conv2D(128, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(128, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)


    layer = Conv2DTranspose(64, 2, strides=2)(layer)
    layer = ReLU()(layer)

    
    layer = Concatenate()([layer, d1])
    layer = Conv2D(64, 1)(layer)
    layer = ReLU()(layer)

    layer = Conv2D(64, 3, padding="same")(layer)
    layer = ReLU()(layer)
    layer = Conv2D(64, 3, padding="same")(layer)
    layer = ReLU()(layer)

    layer = Conv2D(2, 3, activation="tanh", dtype = tf.float32, padding="same")(layer)

    optimizer = tf.optimizers.Nadam()

    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic = True)


    model = tf.keras.Model(inputs = input, outputs = layer, name = model_name)
    model.compile(loss="mse", optimizer=optimizer, metrics=CUSTOM_METRICS)

    return model
    





def make_model_classification(model_name):

    input = tf.keras.Input(shape = INPUT_SIZE_NUMPY)

    layer = input

    #All filters and units should be multiple of 8(even better 128 for TPUs) for efficiency
    layer = Conv2D(32, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(32, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    d1 = ReLU()(layer)

    layer = MaxPool2D()(d1)

    layer = Conv2D(64, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(64, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    d2 = ReLU()(layer)

    layer = MaxPool2D()(d2)
   

    layer = Conv2D(128, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(128, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    d3 = ReLU()(layer)


    layer = MaxPool2D()(d3)

    layer = Conv2D(256, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(256, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    out_encoder = ReLU()(layer)

    


    ##########1
    layer = Conv2DTranspose(128, 3, strides=2)(out_encoder)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    
    layer = Concatenate()([layer, d3])
    layer = Conv2D(128, 1)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = Conv2D(128, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(128, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)


    layer = Conv2DTranspose(64, 3, strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    
    layer = Concatenate()([layer, d2])
    layer = Conv2D(64, 1)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = Conv2D(64, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(64, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)


    layer = Conv2DTranspose(32, 2, strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    
    layer = Concatenate()([layer, d1])
    layer = Conv2D(32, 1)(layer)
    layer = ReLU()(layer)

    layer = Conv2D(32, 3, padding="same")(layer)
    layer = ReLU()(layer)
    layer = Conv2D(32, 3, padding="same")(layer)
    layer = ReLU()(layer)

    out_1 = Conv2D(256, 3, activation="softmax", dtype = tf.float32, padding="same", name="out_1")(layer)


    ################2
    layer = Conv2DTranspose(128, 3, strides=2)(out_encoder)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    
    layer = Concatenate()([layer, d3])
    layer = Conv2D(128, 1)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = Conv2D(128, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(128, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)


    layer = Conv2DTranspose(64, 3, strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    
    layer = Concatenate()([layer, d2])
    layer = Conv2D(64, 1)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = Conv2D(64, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(64, 3, padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)


    layer = Conv2DTranspose(32, 2, strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    
    layer = Concatenate()([layer, d1])
    layer = Conv2D(32, 1)(layer)
    layer = ReLU()(layer)

    layer = Conv2D(32, 3, padding="same")(layer)
    layer = ReLU()(layer)
    layer = Conv2D(32, 3, padding="same")(layer)
    layer = ReLU()(layer)
    #layer = Conv2D(GROUND_TRUTH_SIZE_NUMPY[2], 3, activation="tanh", dtype = tf.float32, padding="same")(layer)

    out_2 = Conv2D(256, 3, activation="softmax", dtype = tf.float32, padding="same", name="out_2")(layer)

    optimizer = tf.optimizers.Nadam()

    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic = True)

    custom_metrics = {"out_1":[c_m for c_m in CUSTOM_METRICS], "out_2":[c_m for c_m in CUSTOM_METRICS]}

    model = tf.keras.Model(inputs = input, outputs = [out_1, out_2], name = model_name)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=custom_metrics)

    return model
    