import keras
import os
import math
import numpy as np
import cv2
from multiprocessing import Pool

from config import INPUT_SIZE, GROUND_TRUTH_SIZE, NR_OF_PROCESSES, INTERPOLATION_RESIZE, RESIZE_TO_TRAINING_SIZE
from exeptions import GroundTruthSizeError
from util import GROUND_TRUTH_SIZE_NUMPY


if GROUND_TRUTH_SIZE[2] not in [2, 3]:
    raise GroundTruthSizeError(GROUND_TRUTH_SIZE)



def load_samples(samples_dir):
    return [os.path.join(samples_dir, filepath) for filepath in os.listdir(samples_dir)]


def read_grayscale_channel_resized_to_fit_network(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)

    height, width = image.shape[0], image.shape[1]

    while width % 2 != 0 or width // 2 % 2 != 1 or width // 2 // 2 % 2 != 1 or width // 2 // 2 // 2 % 2 != 0 or width // 2 // 2 // 2 // 2 % 2 != 1:
        width += 1

    while height % 2 != 0 or height // 2 % 2 != 1 or height // 2 // 2 % 2 != 1 or height // 2 // 2 // 2 % 2 != 0 or width // 2 // 2 // 2 // 2 % 2 != 1:
        height += 1

    return convert_to_grayscale(cv2.resize(image, (width, height)))


def read_grayscale_channel_resized(filepath):
   return convert_to_grayscale(cv2.resize(cv2.imread(filepath, cv2.IMREAD_COLOR), INPUT_SIZE[:-1], interpolation=INTERPOLATION_RESIZE))

def convert_to_grayscale(image):
    return np.reshape(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (*image.shape[:-1], 1))

def read_bgr_channels_resized(filepath):
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

        self.pool = Pool(NR_OF_PROCESSES)

    def __len__(self):
        return math.ceil(len(self.sample_paths) / self.batch_size)

    def __getitem__(self, iteration_n):
        filepaths = self.sample_paths[self.batch_size * iteration_n : self.batch_size * (iteration_n + 1)]
        
        ground_truths = np.array(self.pool.map(read_bgr_channels_resized, filepaths))
        images = np.array(self.pool.map(convert_to_grayscale, ground_truths))

        if GROUND_TRUTH_SIZE[2] == 2:
            ground_truths = np.array(self.pool.map(convert_get_CrCb_channels, ground_truths))

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

        self.pool = Pool(NR_OF_PROCESSES)

    def __len__(self):
        return math.ceil(len(self.sample_paths) / self.batch_size)

    def __getitem__(self, iteration_n):
        filepaths = self.sample_paths[self.batch_size * iteration_n : self.batch_size * (iteration_n + 1)]
        
        if RESIZE_TO_TRAINING_SIZE:
            images = np.array(self.pool.map(read_grayscale_channel_resized, filepaths))
        else:
            images = np.array(self.pool.map(read_grayscale_channel_resized_to_fit_network, filepaths))

        if self.preprocessing_procedure is None:
            return images

        none_list = [None for _ in range(len(images))]
        results = self.pool.starmap(self.preprocessing_procedure, zip(images, none_list))
        images = list(zip(*results))[0]

        images = np.array(images)

        return images




def split_cr_cb(ground_truth):
    cr, cb = cv2.split(ground_truth)

    cr = np.reshape(cr, list(ground_truth.shape[:-1]) + [1])
    cb = np.reshape(cb, list(ground_truth.shape[:-1]) + [1])
    
    return cr, cb




class TrainingGenerator_classification(TrainingGenerator):

    def __getitem__(self, iteration_n):
        filepaths = self.sample_paths[self.batch_size * iteration_n : self.batch_size * (iteration_n + 1)]
        
        ground_truths = np.array(self.pool.map(read_bgr_channels_resized, filepaths))
        images = np.array(self.pool.map(convert_to_grayscale, ground_truths))

        if GROUND_TRUTH_SIZE[2] == 2:
            ground_truths = np.array(self.pool.map(convert_get_CrCb_channels, ground_truths))

        if self.preprocessing_procedure is None:
            return images, ground_truths


        results = self.pool.starmap(self.preprocessing_procedure, zip(images, ground_truths))
        results = list(zip(*results))
        images = results[0]
        ground_truths = results[1]

        images = np.array(images)
        ground_truths = np.array(ground_truths)


        results = list(map(split_cr_cb, ground_truths))
        results = list(zip(*results))
        crs = results[0]
        cbs = results[1]

        crs = np.array(crs)
        cbs = np.array(cbs)

        return images, (crs, cbs)
