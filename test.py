import tensorflow as tf

layer = tf.constant([1, 2, -4])

print(tf.minimum(layer, 1))
