import tensorflow as tf
import numpy as np

from config import NOISE_PERCENTAGE

def preprocess_image_training(image, ground_truth):
    image = tf.cast(image, tf.float32)

    image = image / 255 * 2 - 1

    noise = tf.random.normal(shape=image.shape, mean=0, stddev=1, dtype=tf.float32) * NOISE_PERCENTAGE

    image = image + noise

    image = tf.clip_by_value(image, -1, 1)

    seed = (np.random.randint(0, 10**6, dtype=np.uint32), np.random.randint(0, 10**6, dtype=np.uint32))

    image = tf.image.stateless_random_flip_left_right(image, seed=seed)

    ground_truth = tf.image.stateless_random_flip_left_right(ground_truth, seed=seed)

    return image, ground_truth

def preprocess_image_predicting(image, ground_truth):
    image = tf.cast(image, tf.float32)

    image = image / 255 * 2 - 1

    return image, ground_truth
