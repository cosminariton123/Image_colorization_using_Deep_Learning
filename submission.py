import os

import tensorflow as tf
import cv2

from data_loader import load_samples
from data_loader import PredictionsGenerator
from paths import TEST_SAMPLES_DIR, IMAGE_PREDICTIONS_FOLDER
from preprocessing import preprocess_image_predicting
from custom_metrics import CUSTOM_METRICS
from config import PREDICTION_BATCH_SIZE

def compile_custom_objects():
    custom_objects = dict()

    for custom_metric in CUSTOM_METRICS:
        custom_objects[custom_metric.__name__] = custom_metric
    
    return custom_objects

def load_and_make_submission(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects=compile_custom_objects())

    predicts = model.predict(
        PredictionsGenerator(
            samples_dir=TEST_SAMPLES_DIR,
            batch_size=PREDICTION_BATCH_SIZE,
            preprocessing_procedure=preprocess_image_predicting,
        )
    )


    ids = load_samples(TEST_SAMPLES_DIR)
    ids = [os.path.basename(elem) for elem in ids]

    if not os.path.exists(IMAGE_PREDICTIONS_FOLDER):
        os.mkdir(IMAGE_PREDICTIONS_FOLDER)

    for id, prediction in zip(ids ,predicts):
        cv2.imwrite(os.path.join(IMAGE_PREDICTIONS_FOLDER, id), prediction)
      