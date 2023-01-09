import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, Multiply

from custom_metrics import CUSTOM_METRICS
from config import INPUT_SIZE

def make_model():
    input = tf.keras.Input(shape = INPUT_SIZE)

    layer = input

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
    
    layer = Conv2DTranspose(3, (3, 3), activation="sigmoid")(layer)

    layer = layer * 255


                
    model = tf.keras.Model(inputs = input, outputs = layer)
    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=CUSTOM_METRICS)

    return model
