import tensorflow as tf
import tensorflow_addons as tfa
from preprocessing import unnormalize_pixel_values, normalize_pixel_values

#Manual conversion without opencv so tensorflow doesn't get confused
def ycrcb_to_rgb(ycrcb):
        ycrcb = tf.cast(ycrcb, tf.float32)
        ycrcb = unnormalize_pixel_values(ycrcb)
        ycrcb = ycrcb / 255

        y = ycrcb[:, :, :, 0]
        cr = ycrcb[:, :, :, 1]
        cb = ycrcb[:, :, :, 2]
        
        r = y + 1.403 * (cr - 0.5)
        g = y - 0.714 * (cr - 0.5) - 0.344 * (cb - 0.5)
        b = y + 1.772 * (cb - 0.5)
        
        rgb = tf.concat([r, g, b], axis=-1)

        return rgb

def faint_color_loss(y_true, y_pred):

    y_true_rgb = ycrcb_to_rgb(tf.concat([tf.ones_like(y_true[:,:,:,:0]), y_true], axis=-1))
    y_pred_rgb = ycrcb_to_rgb(tf.concat([tf.ones_like(y_pred[:,:,:,:0]), y_pred], axis=-1))

    y_true_hsv = tf.image.rgb_to_hsv(y_true_rgb)
    y_pred_hsv = tf.image.rgb_to_hsv(y_pred_rgb)

    saturation_loss = tf.losses.mse(y_true_hsv[:,:,:,1], y_pred_hsv[:,:,:,1])
    mse_loss = tf.losses.mse(y_true, y_pred)

    return mse_loss + saturation_loss


def classification_mae(y_true, y_pred):
    y_pred = normalize_pixel_values(tf.argmax(y_pred, axis=-1))
    y_pred = tf.reshape(y_pred, shape=tf.concat([tf.shape(y_pred), tf.constant([1])], axis=0))
    y_true = normalize_pixel_values(y_true)
    
    return tf.losses.mean_absolute_error(y_true, y_pred)


def classification_mse(y_true, y_pred):
    y_pred = normalize_pixel_values(tf.argmax(y_pred, axis=-1))
    y_pred = tf.reshape(y_pred, shape=tf.concat([tf.shape(y_pred), tf.constant([1])], axis=0))
    y_true = normalize_pixel_values(y_true)
    
    return tf.losses.mean_squared_error(y_true, y_pred)

CUSTOM_LOSS = [faint_color_loss]
CUSTOM_METRICS = ["mse", "mae"]#[classification_mse, classification_mae]

CUSTOM_OBJECTS = CUSTOM_LOSS + CUSTOM_METRICS
