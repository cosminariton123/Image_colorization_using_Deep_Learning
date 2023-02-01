import tensorflow as tf

from config import NOISE_PERCENTAGE

def preprocess_image_training(image, label):
    image = tf.cast(image, tf.float32)

    image = image / 255 * 2 - 1

    noise = tf.random.normal(shape=image.shape, mean=0, stddev=0.1, dtype=tf.float32) * tf.floor(tf.random.uniform(shape=image.shape) * (1 + NOISE_PERCENTAGE))

    image = image + noise

    image = tf.maximum(image, -1)
    image = tf.minimum(image, 1)

    return image, label

def preprocess_image_predicting(image, label):
    image = tf.cast(image, tf.float32)

    image = image / 255 * 2 - 1

    return image, label
