import tensorflow as tf

def preprocess_image_training(image, label):
    image = tf.cast(image, tf.float32)

    image = image / 255

    noise = tf.random.normal(shape=image.shape, mean=0, stddev=0.1, dtype=tf.float32)
    image = image + noise

    return image, label

def preprocess_image_predicting(image, label):
    image = tf.cast(image, tf.float32)

    image = image / 255

    return image, label
