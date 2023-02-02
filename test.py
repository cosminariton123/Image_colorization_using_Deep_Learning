from data_loader import read_CrCb_channels, load_samples, read_grayscale_channel, read_rgb_channels
from paths import VALIDATION_SAMPLES_DIR, TRAIN_SAMPLES_DIR

import cv2
import numpy as np



def check_concatenation():

    sample_paths = load_samples(VALIDATION_SAMPLES_DIR)
    #crcb = read_CrCb_channels(sample_paths[0])
    y = read_rgb_channels(sample_paths[0])

    import tensorflow as tf
    cv2.imshow("coco", np.array(tf.image.random_flip_left_right(tf.constant(y), seed=None)))

    #cv2.imshow("coco", cv2.cvtColor(np.concatenate([y, crcb], axis=2), cv2.COLOR_YCrCb2BGR))
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    check_concatenation()

if __name__ == "__main__":
    main()
