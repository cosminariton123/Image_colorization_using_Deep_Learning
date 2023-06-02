import os
import shutil

from keras import mixed_precision
from limit_gpu_memory_growth import limit_gpu_memory_growth

from model_generator import make_model
from tunning import train_model_and_save
from prediction import load_and_make_prediction, load_model, load_and_make_prediction_best_model

from paths import OUTPUT_DIR, TEST_SAMPLES_DIR, VALIDATION_SAMPLES_DIR

from util import get_saved_model_epoch_as_int_from_filename

from config import LIMIT_GPU_MEMORY_GROWTH, MIXED_PRECISION_16


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

    if LIMIT_GPU_MEMORY_GROWTH is True: limit_gpu_memory_growth()
    
    if MIXED_PRECISION_16 is True: mixed_precision.set_global_policy('mixed_float16')

    model_name = "deep_high_filters_UnetCNN"

    train_model_from_scratch(model_name)
    #load_and_train_model(os.path.join(OUTPUT_DIR, model_name, "model_saves", "model_epoch_1.hdf5"))
    #load_and_make_prediction(os.path.join(OUTPUT_DIR, model_name, "model_saves", "best_model_epoch_81.hdf5"), VALIDATION_SAMPLES_DIR)
    #load_and_make_prediction(os.path.join(OUTPUT_DIR, model_name, "model_saves", "model_epoch_143.hdf5"), VALIDATION_SAMPLES_DIR)
    load_and_make_prediction_best_model(os.path.join(OUTPUT_DIR, model_name), VALIDATION_SAMPLES_DIR)

    def train_models(models_names):
        from multiprocessing import Process

        processes = list()
        paths = [os.path.join(OUTPUT_DIR, model_name, "model_saves", "model_epoch_1.hdf5") for model_name in models_names]
        
        for path in paths:
            p = Process(target=load_and_train_model, args=(path,))
            processes.append(p)
        
        for p in processes:
            p.start()
            p.join()

    def train_models_rgb(models_names):
        from multiprocessing import Process

        processes = list()
        paths = [os.path.join(OUTPUT_DIR, model_name, "model_saves", "model_epoch_1.hdf5") for model_name in models_names]
        
        for path in paths:
            p = Process(target=load_and_train_model, args=(path,))
            processes.append(p)
        
        for p in processes:
            p.start()
            p.join()

    def predict_models(models_names):
        from multiprocessing import Process

        processes = list()
        paths = paths = paths = [os.path.join(OUTPUT_DIR, model_name) for model_name in models_names]
        
        for path in paths:
            p = Process(target=load_and_make_prediction_best_model, args=(path, VALIDATION_SAMPLES_DIR))
            processes.append(p)
        
        for p in processes:
            p.start()
            p.join()

    def predict_models_rgb(models_names):
        from multiprocessing import Process

        processes = list()
        paths = [os.path.join(OUTPUT_DIR, model_name) for model_name in models_names]
        
        for path in paths:
            p = Process(target=load_and_make_prediction_best_model, args=(path, VALIDATION_SAMPLES_DIR))
            processes.append(p)
        
        for p in processes:
            p.start()
            p.join()


    models_names = [elem for elem in os.listdir(OUTPUT_DIR) if "rgb" not in elem]
    models_names_rgb = [elem for elem in os.listdir(OUTPUT_DIR) if "rgb" in elem]

    #print(f"Training {len(models_names)} ycrcb models")
    #train_models(models_names)
    #predict_models(models_names)

    #print(f"Training {len(models_names_rgb)} rgb models")
    #train_models_rgb(models_names_rgb)
    #predict_models_rgb(models_names_rgb)

if __name__ == "__main__":
    main()
