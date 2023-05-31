import numpy as np
import cv2
import random #RANDOM IS FASTER THAN NUMPY FOR SINGLE SAMPLE

from config import NOISE_PERCENTAGE, NOISE_DEVIATION, NOISE_MEAN, MAX_DEGREE_OF_LEFT_RIGHT_ROTATION, INTERPOLATION_ROTATE
from exeptions import NoiseDeviationValueError, NoisePercentageValueError

if NOISE_PERCENTAGE < 0 or NOISE_PERCENTAGE > 1:
    raise NoisePercentageValueError(NOISE_PERCENTAGE)

if NOISE_DEVIATION < 0 or NOISE_DEVIATION > 999:
    raise NoiseDeviationValueError(NOISE_DEVIATION)


def add_noise(image):
    noise = np.random.normal(NOISE_MEAN, NOISE_DEVIATION, image.shape) * np.floor(np.random.uniform(size=image.shape) * (1 + NOISE_PERCENTAGE))
    image = image + noise
    image = np.clip(image, 0, 255)
    return image

def random_flip_left_right(image, ground_truth):
    if random.choice([True, False]):
        image = cv2.flip(image, 1)
        ground_truth = cv2.flip(ground_truth, 1)

    return image, ground_truth

def random_rotate(image, ground_truth):
    height, width = image.shape[0], image.shape[1]
    degrees = random.uniform(0, MAX_DEGREE_OF_LEFT_RIGHT_ROTATION)
    degrees = -degrees if random.choice([True, False]) else degrees
    scale = 1

    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, degrees, scale)
    image_rotated = cv2.warpAffine(image, rotation_matrix, (width, height), flags=INTERPOLATION_ROTATE)
    ground_truth_rotated = cv2.warpAffine(ground_truth, rotation_matrix, (width, height), flags=INTERPOLATION_ROTATE)

    return image_rotated, ground_truth_rotated


def normalize_pixel_values(image):
    return image / 255 * 2 - 1


def unnormalize_pixel_values(image):
    return (image + 1) / 2 * 255


def preprocess_image_training(image, ground_truth):
    image_shape = image.shape
    ground_truth_shape = ground_truth.shape

    image = np.array(image, dtype=np.float32)

    image = add_noise(image)
    image, ground_truth = random_flip_left_right(image, ground_truth)
    image, ground_truth = random_rotate(image, ground_truth)
    image = normalize_pixel_values(image)
    ground_truth = normalize_pixel_values(ground_truth)

    image = np.reshape(image, image_shape)
    ground_truth = np.reshape(ground_truth, ground_truth_shape)

    return image, ground_truth



def preprocess_image_predicting(image, ground_truth):
    image = np.array(image, dtype=np.float32)

    image = normalize_pixel_values(image)
    if ground_truth is not None:
        ground_truth = normalize_pixel_values(ground_truth)

    return image, ground_truth
