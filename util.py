import os
import re

from config import INPUT_SIZE, GROUND_TRUTH_SIZE, INTERPOLATION_ROTATE

from PIL import Image
import math
import numpy as np

def max_crop_dimensions(height, width, degree):

    sin_angle = abs(math.sin(degree))
    cos_angle = abs(math.cos(degree))

    if height < width:
        L = width
        l = height
    else:
        L = height
        l = width

    if l <= L * 2 * sin_angle * cos_angle:
        if height < width:
            height = (l / 2) / cos_angle
            width = (l / 2) / sin_angle
        else:
            height = (l / 2) / sin_angle
            width = (l / 2) / cos_angle
    else:
        aux_width = width
        width = (width * cos_angle - height * sin_angle) / (cos_angle * cos_angle - sin_angle * sin_angle)
        height = (height * cos_angle - aux_width * sin_angle) / (cos_angle * cos_angle - sin_angle * sin_angle)
        
    return height, width




def rotate_and_crop(image, degrees):
    
    optimum_height, optimum_width = max_crop_dimensions(image.shape[0], image.shape[1], math.radians(degrees))

    image_rotated = np.array(Image.fromarray(image).rotate(-degrees, expand=True, resample=INTERPOLATION_ROTATE))
    
    height, width = image_rotated.shape[0], image_rotated.shape[1]

    return image_rotated[int((height - optimum_height) / 2):int((height - optimum_height) / 2 + optimum_height), int((width - optimum_width) /2):int((width - optimum_width) / 2 + optimum_width)]





def convert_from_image_to_numpy_notation(size_tuple):
    size = list(size_tuple)
    size =  [size[1], size[0]] + size[2:]
    
    return tuple(size)

def compute_input_size_numpy():
    return convert_from_image_to_numpy_notation(INPUT_SIZE)

def compute_ground_truth_size_numpy():
    return convert_from_image_to_numpy_notation(GROUND_TRUTH_SIZE)


def generate_log_dir_of_not_exists(save_path):
    log_dir = os.path.join(save_path, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    return log_dir


def get_saved_model_epoch_as_int_from_filename(filename):
    return int(filename.split("_")[-1].split(".")[0])


def get_all_saved_models_from_dir_without_best(path):
    files = os.listdir(path)
    files = [file for file in files if re.fullmatch(re.compile("^model_epoch_[0-9]+\.hdf5$") , file)]
    return files


def get_all_best_saved_models_from_dir(path):
    files = os.listdir(path)
    files = [file for file in files if re.fullmatch(re.compile("^best_model_epoch_[0-9]+\.hdf5$") , file)]
    return files


def remove_old_models_until_x_most_recent_remain(epoch, model_saves_dir, how_many_remain):
    epoch = epoch + 1
    files = get_all_saved_models_from_dir_without_best(model_saves_dir)

    files = [file for file in files if get_saved_model_epoch_as_int_from_filename(file) > epoch or epoch - how_many_remain >= get_saved_model_epoch_as_int_from_filename(file)]

    for file in files:
        filepath = os.path.join(model_saves_dir, file)
        os.remove(filepath)
        print(f"Epoch {epoch}: deleting old model {filepath}")


def remove_old_best_models(epoch, model_saves_dir):
    epoch = epoch + 1
    files = get_all_best_saved_models_from_dir(model_saves_dir)

    for file in files:
        if get_saved_model_epoch_as_int_from_filename(file) > epoch:
            filepath = os.path.join(model_saves_dir, file)
            os.remove(filepath)
            print(f"Epoch {epoch} deleting old best model {filepath}")

    if len(files) > 1:
        for file in files:
            if get_saved_model_epoch_as_int_from_filename(file) != epoch:
                filepath = os.path.join(model_saves_dir, file)
                os.remove(filepath)
                print(f"Epoch {epoch} deleting old best model {filepath}")


INPUT_SIZE_NUMPY = compute_input_size_numpy()

GROUND_TRUTH_SIZE_NUMPY = compute_ground_truth_size_numpy()
