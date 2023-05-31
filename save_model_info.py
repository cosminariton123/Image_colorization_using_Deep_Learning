import math
import numpy as np
import cv2

import tensorflow as tf
from keras.models import Model

from config import TRAINING_IMAGE_DIFFERENT_SAMPLES_TO_LOG, TRAINING_IMAGE_SAME_SAMPLE_TO_LOG, VALIDATION_IMAGE_SAMPLES_TO_LOG, GROUND_TRUTH_SIZE

from preprocessing import unnormalize_pixel_values


def generate_summary(model: Model):
    summary = []
    model.summary(print_fn=lambda line: summary.append(line))
    summary = "\n".join(summary)
    
    return summary


def log_summary(string, log_dir):
    file_writer = tf.summary.create_file_writer(log_dir)

    string = "".join([f"    {line}\n" for line in string.split("\n")])

    with file_writer.as_default():
        tf.summary.text("Summary", string, step=0)


def generate_print_and_log_summary(model: Model, save_path):
    summary = generate_summary(model)
    print(summary)
    log_summary(summary, save_path)



def log_image_samples_at_the_start_of_training(log_dir, initial_epoch, training_data_generator, validation_data_generator):
    batch_size = len(training_data_generator[0])
    how_many_batches = math.ceil(TRAINING_IMAGE_DIFFERENT_SAMPLES_TO_LOG / batch_size)

    training_input_images_to_log = list()
    training_ground_truth_images_to_log = list()

    images_counter = 0
    for batch_idx in range(how_many_batches):
        batch_resamples = list()
        for _ in range(TRAINING_IMAGE_SAME_SAMPLE_TO_LOG):
            batch_resamples.append(training_data_generator[batch_idx])

        for i in range(batch_size):
            for batch in batch_resamples:
                if images_counter >= TRAINING_IMAGE_DIFFERENT_SAMPLES_TO_LOG:
                    break

                training_input_images_to_log.append(batch[0][i])
                training_ground_truth_images_to_log.append(batch[1][i])


    training_input_images_to_log = np.array(unnormalize_pixel_values(np.array(training_input_images_to_log)), dtype=np.uint8)
    training_ground_truth_images_to_log = np.array(unnormalize_pixel_values(np.array(training_ground_truth_images_to_log)), dtype=np.uint8)

    validation_input_images_to_log = list()
    validation_ground_truth_images_to_log = list()

    images_counter = 0
    for batch_idx in range(how_many_batches):
        batch = validation_data_generator[batch_idx]
        for i in range(len(batch[0])):
            if images_counter >= VALIDATION_IMAGE_SAMPLES_TO_LOG:
                break
            
            validation_input_images_to_log.append(batch[0][i])
            validation_ground_truth_images_to_log.append(batch[1][i])

    validation_input_images_to_log = np.array(unnormalize_pixel_values(np.array(validation_input_images_to_log)), dtype=np.uint8)
    validation_ground_truth_images_to_log = np.array(unnormalize_pixel_values(np.array(validation_ground_truth_images_to_log)), dtype=np.uint8)


    if GROUND_TRUTH_SIZE[2] == 2:
        training_ground_truth_images_to_log = np.array([cv2.cvtColor(np.concatenate([input, ground_truth], axis=2, dtype=np.uint8), cv2.COLOR_YCrCb2RGB) for input, ground_truth in zip(training_input_images_to_log, training_ground_truth_images_to_log)])
        validation_ground_truth_images_to_log = np.array([cv2.cvtColor(np.concatenate([input, ground_truth], axis=2, dtype=np.uint8), cv2.COLOR_YCrCb2RGB) for input, ground_truth in zip(validation_input_images_to_log, validation_ground_truth_images_to_log)])
    else:
        training_ground_truth_images_to_log = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in training_ground_truth_images_to_log])
        validation_ground_truth_images_to_log = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in validation_ground_truth_images_to_log])

    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
        tf.summary.image("Training input data samples", training_input_images_to_log, step=initial_epoch, max_outputs=TRAINING_IMAGE_DIFFERENT_SAMPLES_TO_LOG * TRAINING_IMAGE_SAME_SAMPLE_TO_LOG)
        tf.summary.image("Training ground truth data samples", training_ground_truth_images_to_log, step=initial_epoch, max_outputs=TRAINING_IMAGE_DIFFERENT_SAMPLES_TO_LOG * TRAINING_IMAGE_SAME_SAMPLE_TO_LOG)
        tf.summary.image("Validation input data samples", validation_input_images_to_log, step=initial_epoch, max_outputs=VALIDATION_IMAGE_SAMPLES_TO_LOG)
        tf.summary.image("Validation ground truth data samples", validation_ground_truth_images_to_log, step=initial_epoch, max_outputs=VALIDATION_IMAGE_SAMPLES_TO_LOG)
