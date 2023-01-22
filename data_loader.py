import keras
import os
import math
import numpy as np
import cv2

from config import INPUT_SIZE, GROUND_THRUTH_SIZE


def load_samples(samples_dir):
    return [os.path.join(samples_dir, filepath) for filepath in os.listdir(samples_dir)]


class TrainingGenerator(keras.utils.Sequence):
    def __init__(self, samples_dir, batch_size, preprocessing_procedure, shuffle = True):
        self.sample_paths = np.array(load_samples(samples_dir))

        self.preprocessing_procedure = preprocessing_procedure

        self.shuffle = shuffle

        if self.shuffle:
            np.random.shuffle(self.sample_paths)
        
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.sample_paths) / self.batch_size)

    def __getitem__(self, iteration_n):
        filepaths = self.sample_paths[self.batch_size * iteration_n : self.batch_size * (iteration_n + 1)]
        
        samples = np.array([np.reshape(cv2.resize(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), INPUT_SIZE[:-1]), (INPUT_SIZE[1], INPUT_SIZE[0], INPUT_SIZE[2])) for filepath in filepaths])
        ground_thruths = np.array([np.reshape(cv2.resize(cv2.imread(filepath, cv2.IMREAD_COLOR), GROUND_THRUTH_SIZE[:-1]), (GROUND_THRUTH_SIZE[1], GROUND_THRUTH_SIZE[0], GROUND_THRUTH_SIZE[2])) for filepath in filepaths])

        
        preprocessed_samples = list()
        preprocessed_labels = list()
        for elem_data, elem_label in zip(samples, ground_thruths):
            preprocessed_elem_data, preprocessed_elem_label = self.preprocessing_procedure(elem_data, elem_label)
            preprocessed_samples.append(preprocessed_elem_data)
            preprocessed_labels.append(preprocessed_elem_label)
        preprocessed_samples = np.array(preprocessed_samples)
        preprocessed_labels = np.array(preprocessed_labels)

        return preprocessed_samples, preprocessed_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.sample_paths)


class PredictionsGenerator(keras.utils.Sequence):
    def __init__(self, samples_dir, batch_size, preprocessing_procedure):
        self.sample_paths = np.array(load_samples(samples_dir))

        self.preprocessing_procedure = preprocessing_procedure
        
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.sample_paths) / self.batch_size)

    def __getitem__(self, iteration_n):
        filepaths = self.sample_paths[self.batch_size * iteration_n : self.batch_size * (iteration_n + 1)]
        
        samples = np.array([np.reshape(cv2.resize(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), INPUT_SIZE[:-1]), (INPUT_SIZE[1], INPUT_SIZE[0], INPUT_SIZE[2])) for filepath in filepaths])

        preprocessed_samples = list()
        for elem_data in samples:
            preprocessed_elem_data, _ = self.preprocessing_procedure(elem_data, None)
            preprocessed_samples.append(preprocessed_elem_data)
        preprocessed_samples = np.array(preprocessed_samples)

        return preprocessed_samples
