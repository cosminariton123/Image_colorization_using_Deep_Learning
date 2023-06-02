import os
import cv2
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from keras.models import Model

from data_loader import load_samples
from data_loader import PredictionsGenerator
from preprocessing import preprocess_image_predicting, unnormalize_pixel_values

from custom_metrics import CUSTOM_METRICS
from config import PREDICTION_BATCH_SIZE, GROUND_TRUTH_SIZE
from exeptions import GroundTruthSizeError

def compile_custom_objects():
    custom_objects = dict()

    for custom_metric in CUSTOM_METRICS:
        custom_objects[custom_metric.__name__] = custom_metric
    
    return custom_objects


def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects=compile_custom_objects())
    

def make_prediction(model:Model, input_dir, output_dir):
    inputs_as_batches = (
        PredictionsGenerator(
            samples_dir=input_dir,
            batch_size=PREDICTION_BATCH_SIZE,
            preprocessing_procedure=preprocess_image_predicting,
        )
    )


    raw_inputs_as_batches = (
        PredictionsGenerator(
            samples_dir=input_dir,
            batch_size=PREDICTION_BATCH_SIZE,
        )
    )


    ids = load_samples(input_dir)
    ids = [os.path.basename(elem) for elem in ids]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    id_idx = 0
    for inputs_as_batch, raw_inputs_as_batch in tqdm(zip(inputs_as_batches, raw_inputs_as_batches), desc="Predicting image batches", total=len(inputs_as_batches)):
        predictions = np.array(model(inputs_as_batch, training=False))

        if GROUND_TRUTH_SIZE[2] == 3:
            for prediction in predictions:
                cv2.imwrite(os.path.join(output_dir, ids[id_idx]), np.array(unnormalize_pixel_values(prediction), dtype=np.uint8))
                id_idx += 1
        elif GROUND_TRUTH_SIZE[2] == 2:
            for prediction, raw_input in zip(predictions, raw_inputs_as_batch):
                cv2.imwrite(os.path.join(output_dir, ids[id_idx]) , cv2.cvtColor(np.concatenate([np.array(raw_input, dtype=np.uint8), np.array(unnormalize_pixel_values(prediction), dtype=np.uint8)], axis=2, dtype=np.uint8), cv2.COLOR_YCrCb2BGR))
                id_idx += 1
        else:
            raise GroundTruthSizeError(GROUND_TRUTH_SIZE)


def load_and_make_prediction(model_path, input_dir):
    model_dir = os.path.dirname(os.path.dirname(model_path))

    model = load_model(model_path)
    make_prediction(model, input_dir, os.path.join(model_dir, "Image Predictions"))


def load_and_make_prediction_best_model(model_dir, input_dir):
    models_paths = [os.path.join(model_dir, "model_saves", elem) for elem in os.listdir(os.path.join(model_dir, "model_saves"))]

    best_model_path = [model_path for model_path in models_paths if "best" in model_path][0]
    
    model = load_model(best_model_path)
    make_prediction(model, input_dir, os.path.join(model_dir, "Image Predictions"))