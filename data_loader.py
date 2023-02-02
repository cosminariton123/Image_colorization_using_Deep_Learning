import keras
import os
import math
import numpy as np
import cv2
from multiprocessing import Pool

from config import INPUT_SIZE, GROUND_THRUTH_SIZE, NR_OF_PROCESSES_PER_GENERATOR
from util import INPUT_SIZE_NUMPY, GROUND_TRUTH_SIZE_NUMPY
from exeptions import GroundTruthSizeError


def load_samples(samples_dir):
    return [os.path.join(samples_dir, filepath) for filepath in os.listdir(samples_dir)]


def read_grayscale_channel(filepath):
   return np.reshape(cv2.resize(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), INPUT_SIZE[:-1]), INPUT_SIZE_NUMPY)

def read_rgb_channels(filepath):
    return np.reshape(cv2.resize(cv2.imread(filepath, cv2.IMREAD_COLOR), GROUND_THRUTH_SIZE[:-1]), GROUND_TRUTH_SIZE_NUMPY)

def read_CrCb_channels(filepath):
    return np.reshape(cv2.cvtColor(cv2.resize(cv2.imread(filepath, cv2.IMREAD_COLOR), GROUND_THRUTH_SIZE[:-1]), cv2.COLOR_BGR2YCrCb)[:,:,1:3], GROUND_TRUTH_SIZE_NUMPY)


class TrainingGenerator(keras.utils.Sequence):
    def __init__(self, samples_dir, batch_size, preprocessing_procedure = None, shuffle = True):
        self.sample_paths = np.array(load_samples(samples_dir))

        self.preprocessing_procedure = preprocessing_procedure

        self.shuffle = shuffle

        if self.shuffle:
            np.random.shuffle(self.sample_paths)
        
        self.batch_size = batch_size

        self.pool = Pool(NR_OF_PROCESSES_PER_GENERATOR)

    def __len__(self):
        return math.ceil(len(self.sample_paths) / self.batch_size)

    def __getitem__(self, iteration_n):
        filepaths = self.sample_paths[self.batch_size * iteration_n : self.batch_size * (iteration_n + 1)]
        
        samples = np.array(self.pool.map(read_grayscale_channel, filepaths))

        if GROUND_THRUTH_SIZE[2] == 3:
            ground_thruths = np.array(self.pool.map(read_rgb_channels, filepaths))
        elif GROUND_THRUTH_SIZE[2] == 2:
            ground_thruths = np.array(self.pool.map(read_CrCb_channels, filepaths))
        else:
            raise GroundTruthSizeError(GROUND_THRUTH_SIZE)

        if self.preprocessing_procedure is None:
            return samples, ground_thruths

        preprocessed_samples = list()
        preprocessed_ground_truths = list()
        for elem_data, elem_label in zip(samples, ground_thruths):
            preprocessed_elem_data, preprocessed_elem_label = self.preprocessing_procedure(elem_data, elem_label)
            preprocessed_samples.append(preprocessed_elem_data)
            preprocessed_ground_truths.append(preprocessed_elem_label)
        preprocessed_samples = np.array(preprocessed_samples)
        preprocessed_ground_truths = np.array(preprocessed_ground_truths)

        return preprocessed_samples, preprocessed_ground_truths

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.sample_paths)


class PredictionsGenerator(keras.utils.Sequence):
    def __init__(self, samples_dir, batch_size, preprocessing_procedure = None):
        self.sample_paths = np.array(load_samples(samples_dir))

        self.preprocessing_procedure = preprocessing_procedure
        
        self.batch_size = batch_size

        self.pool = Pool(NR_OF_PROCESSES_PER_GENERATOR)

    def __len__(self):
        return math.ceil(len(self.sample_paths) / self.batch_size)

    def __getitem__(self, iteration_n):
        filepaths = self.sample_paths[self.batch_size * iteration_n : self.batch_size * (iteration_n + 1)]
        
        samples = np.array(self.pool.map(read_grayscale_channel, filepaths))

        if self.preprocessing_procedure is None:
            return samples

        preprocessed_samples = list()
        for elem_data in samples:
            preprocessed_elem_data, _ = self.preprocessing_procedure(elem_data, None)
            preprocessed_samples.append(preprocessed_elem_data)
        preprocessed_samples = np.array(preprocessed_samples)

        return preprocessed_samples
