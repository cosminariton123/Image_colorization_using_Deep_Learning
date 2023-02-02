import tensorflow as tf

from config import NOISE_PERCENTAGE

def preprocess_image_training(image, ground_truth):
    image = tf.cast(image, tf.float32)

    image = image / 255 * 2 - 1

    noise = tf.random.normal(shape=image.shape, mean=0, stddev=1, dtype=tf.float32) * NOISE_PERCENTAGE

    image = image + noise

    image = tf.clip_by_value(image, -1, 1)

    image = tf.image.random_flip_left_right(image, seed=None)

    return image, ground_truth

def preprocess_image_predicting(image, ground_truth):
    image = tf.cast(image, tf.float32)

    image = image / 255 * 2 - 1

    return image, ground_truth
