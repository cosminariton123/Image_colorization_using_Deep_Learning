import os
import shutil

from model_generator import make_model
from tunning import train_model_and_save
from submission import load_and_make_submission, load_model
from paths import OUTPUT_DIR
from limit_gpu_memory_growth import limit_gpu_memory_growth
from keras import mixed_precision


def train_model_from_scratch(model_name):
    model = make_model()

    this_model_path = os.path.join(OUTPUT_DIR, model_name)

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if os.path.exists(this_model_path):
        shutil.rmtree(this_model_path)

    os.mkdir(this_model_path)

    train_model_and_save(model, this_model_path)


def load_and_train_model(model_path):
    model = load_model(model_path)
    train_model_and_save(model, model_path)
    


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

    train_model_from_scratch(model_name)
    load_and_make_submission(os.path.join(OUTPUT_DIR, model_name))


if __name__ == "__main__":
    main()
