import os
import re

from config import INPUT_SIZE, GROUND_TRUTH_SIZE

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


def log_image_samples_at_the_start_of_training(log_dir, training_data_generator, validation_data_generator):
    import matplotlib.pyplot as plt
    import cv2
    for i in range(10):                 #batch black/color #image_in_batch
        cv2.imshow("coco", training_data_generator[0][1][i])
        cv2.waitKey()
        cv2.destroyAllWindows()
        plt.show()

    exit()
 
    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
        tf.summary.image("Training data samples", img, step=0)


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
