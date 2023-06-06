import tensorflow as tf

def ycrcb_to_rgb(ycrcb):
        ycrcb = tf.cast(ycrcb, tf.float32)
        from preprocessing import unnormalize_pixel_values
        ycrcb = unnormalize_pixel_values(ycrcb)
        ycrcb = ycrcb / 255

        y = ycrcb[:, :, :, 0:1]
        cr = ycrcb[:, :, :, 1:2]
        cb = ycrcb[:, :, :, 2:3]
        
        r = y + 1.403 * (cr - 0.5)
        g = y - 0.714 * (cr - 0.5) - 0.344 * (cb - 0.5)
        b = y + 1.772 * (cb - 0.5)
        
        rgb = tf.concat([r, g, b], axis=-1)

        return rgb

def faint_color_loss(y_true, y_pred):
    y_true_reshaped = tf.reshape(y_true, shape=(-1, tf.shape(y_true)[1], tf.shape(y_true)[2], 2))
    y_pred_reshaped = tf.reshape(y_pred, shape=(-1, tf.shape(y_pred)[1], tf.shape(y_pred)[2], 2))

    y_true_rgb = ycrcb_to_rgb(tf.concat([tf.ones_like(y_true_reshaped[:, :, :, :1]), y_true_reshaped], axis=-1))
    y_pred_rgb = ycrcb_to_rgb(tf.concat([tf.ones_like(y_pred_reshaped[:, :, :, :1]), y_pred_reshaped], axis=-1))

    y_true_hsv = tf.image.rgb_to_hsv(y_true_rgb)
    y_pred_hsv = tf.image.rgb_to_hsv(y_pred_rgb)

    perceptual_loss = tf.reduce_mean(tf.square(y_pred_hsv[:, :, :, 1] - y_true_hsv[:, :, :, 1]))

    weighted_perceptual_loss = 1.0 * perceptual_loss

    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    total_loss = mse_loss + weighted_perceptual_loss

    return total_loss

CUSTOM_LOSS = [faint_color_loss]
CUSTOM_METRICS = list()

CUSTOM_OBJECTS = CUSTOM_LOSS + CUSTOM_METRICS
