import os

import tensorflow as tf
from keras.models import Sequential
import cv2
import numpy as np
from tqdm import tqdm

from data_loader import load_samples
from data_loader import PredictionsGenerator
from paths import TEST_SAMPLES_DIR
from preprocessing import preprocess_image_predicting
from custom_metrics import CUSTOM_METRICS
from config import PREDICTION_BATCH_SIZE

def compile_custom_objects():
    custom_objects = dict()

    for custom_metric in CUSTOM_METRICS:
        custom_objects[custom_metric.__name__] = custom_metric
    
    return custom_objects


def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects=compile_custom_objects())
    

def make_submission(model:Sequential, output_dir):
    inputs = (
        PredictionsGenerator(
            samples_dir=TEST_SAMPLES_DIR,
            batch_size=PREDICTION_BATCH_SIZE,
            preprocessing_procedure=preprocess_image_predicting,
        )
    )


    ids = load_samples(TEST_SAMPLES_DIR)
    ids = [os.path.basename(elem) for elem in ids]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    assert len(ids) == PREDICTION_BATCH_SIZE * len(inputs), f"The number of ids and the number of files should be equal.\nNumber of ids: {len(ids)}\nNumber of files: {PREDICTION_BATCH_SIZE * len(inputs)}"

    id_idx = 0
    for input in tqdm(inputs, desc="Predicting image batches", total=len(inputs)):
        predictions = np.array(model(input, training=False))

        for prediction in predictions:
            cv2.imwrite(os.path.join(output_dir, ids[id_idx]), prediction)
            id_idx += 1


def load_and_make_submission(model_path):
    model = load_model(model_path)
    make_submission(model, os.path.join(os.path.dirname(model_path), "Image Predictions"))

    
      