import keras
import os
import math
import numpy as np
import cv2
from multiprocessing import Pool

from config import INPUT_SIZE, GROUND_TRUTH_SIZE, NR_OF_PROCESSES_PER_GENERATOR, INTERPOLATION_RESIZE
from exeptions import GroundTruthSizeError


if GROUND_TRUTH_SIZE[2] not in [2, 3]:
    raise GroundTruthSizeError(GROUND_TRUTH_SIZE)



def load_samples(samples_dir):
    return [os.path.join(samples_dir, filepath) for filepath in os.listdir(samples_dir)]


def read_grayscale_channel(filepath):
   return convert_to_grayscale(cv2.resize(cv2.imread(filepath, cv2.IMREAD_COLOR), INPUT_SIZE[:-1], interpolation=INTERPOLATION_RESIZE))

def convert_to_grayscale(image):
    return np.reshape(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (*image.shape[:-1], 1))

def read_bgr_channels(filepath):
    return cv2.resize(cv2.imread(filepath, cv2.IMREAD_COLOR), GROUND_TRUTH_SIZE[:-1], interpolation=INTERPOLATION_RESIZE)

def convert_get_CrCb_channels(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:,:,1:3]



class TrainingGenerator(keras.utils.Sequence):
    def __init__(self, samples_dir, batch_size, preprocessing_procedure = None, shuffle = True):
        self.sample_paths = np.array(load_samples(samples_dir))

        self.preprocessing_procedure = preprocessing_procedure

        self.shuffle = shuffle

        if self.shuffle:
            np.random.shuffle(self.sample_paths)
        
        self.batch_size = batch_size

        #self.pool = Pool(NR_OF_PROCESSES_PER_GENERATOR)

    def __len__(self):
        return math.ceil(len(self.sample_paths) / self.batch_size)

    def __getitem__(self, iteration_n):
        filepaths = self.sample_paths[self.batch_size * iteration_n : self.batch_size * (iteration_n + 1)]
        pool = Pool(NR_OF_PROCESSES_PER_GENERATOR)
        
        ground_truths = np.array(pool.map(read_bgr_channels, filepaths))
        images = np.array(pool.map(convert_to_grayscale, ground_truths))

        if GROUND_TRUTH_SIZE[2] == 2:
            ground_truths = np.array(pool.map(convert_get_CrCb_channels, ground_truths))

        if self.preprocessing_procedure is None:
            return images, ground_truths


        results = self.pool.starmap(self.preprocessing_procedure, zip(images, ground_truths))
        results = list(zip(*results))
        images = results[0]
        ground_truths = results[1]

        images = np.array(images)
        ground_truths = np.array(ground_truths)

        return images, ground_truths

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.sample_paths)


class PredictionsGenerator(keras.utils.Sequence):
    def __init__(self, samples_dir, batch_size, preprocessing_procedure = None):
        self.sample_paths = np.array(load_samples(samples_dir))

        self.preprocessing_procedure = preprocessing_procedure
        
        self.batch_size = batch_size

        #self.pool = Pool(NR_OF_PROCESSES_PER_GENERATOR)

    def __len__(self):
        return math.ceil(len(self.sample_paths) / self.batch_size)

    def __getitem__(self, iteration_n):
        filepaths = self.sample_paths[self.batch_size * iteration_n : self.batch_size * (iteration_n + 1)]
        pool = Pool(NR_OF_PROCESSES_PER_GENERATOR)
        
        images = np.array(pool.map(read_grayscale_channel, filepaths))

        if self.preprocessing_procedure is None:
            return images

        none_list = [None for _ in range(len(images))]
        results = np.array(pool.starmap(self.preprocessing_procedure, zip(images, none_list)))
        images = list(zip(*results))[0]

        images = np.array(images)

        return images
