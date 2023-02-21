import os
import shutil
from keras import mixed_precision

from model_generator import make_model
from tunning import train_model_and_save
from prediction import load_and_make_prediction, load_model
from paths import OUTPUT_DIR, TEST_SAMPLES_DIR, VALIDATION_SAMPLES_DIR
from limit_gpu_memory_growth import limit_gpu_memory_growth
from util import get_saved_model_epoch_as_int_from_filename


def train_model_from_scratch(model_name):
    model = make_model(model_name)

    save_path = os.path.join(OUTPUT_DIR, model_name)

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    os.mkdir(save_path)

    train_model_and_save(model, save_path)


def load_and_train_model(model_path):
    '''
    Warning: Deletes all saved models after epoch x including best models
    '''

    initial_epoch = get_saved_model_epoch_as_int_from_filename(model_path)

    model_saves_dir = os.path.dirname(model_path)
    save_path = os.path.dirname(model_saves_dir)

    model = load_model(model_path)
    train_model_and_save(model, save_path, initial_epoch)
    


def main():
    #Use this if you want to use your computer for something else
    #and performance is hindered by training
    #limit_gpu_memory_growth()
    
    #Use this if you want to have lower floating point precission(has almost no effect on loss),
    #but use less memory 
    #and compute faster for graphics cards with compute capability above 7
    mixed_precision.set_global_policy('mixed_float16')
    #Same, but for TPUs:
    #mixed_precision.set_global_policy("mixed_bfloat16")

    model_name = "modelCNN"

    #train_model_from_scratch(model_name)
    #load_and_train_model(os.path.join(OUTPUT_DIR, model_name, "model_saves", "model_epoch_7.hdf5"))
    #load_and_make_prediction(os.path.join(OUTPUT_DIR, model_name, "model_saves", "best_model_epoch_10.hdf5"), VALIDATION_SAMPLES_DIR)


if __name__ == "__main__":
    main()
