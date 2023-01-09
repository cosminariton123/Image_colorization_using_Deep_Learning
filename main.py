import os

from model_generator import make_model
from tunning import search_for_best_model_and_save
from submission import load_and_make_submission
from paths import OUTPUT_DIR
    

def train_model():
    #Use this if you want to use your computer for something else
    #and performance is hindered by training
    #limit_gpu_memory_growth()

    model = make_model()

    this_model_path = os.path.join(OUTPUT_DIR, "modelCNN")

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if not os.path.exists(this_model_path):
        os.mkdir(this_model_path)
    search_for_best_model_and_save(model, this_model_path)

def main():
    
    train_model()
    load_and_make_submission(os.path.join(OUTPUT_DIR, "modelCNN", "best_model.hdf5"))


if __name__ == "__main__":
    main()
