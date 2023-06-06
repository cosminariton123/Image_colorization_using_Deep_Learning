import os
import shutil
from tqdm import tqdm
from multiprocessing import Process

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

    #model_name = "5ks_deep_high_filters_UnetCNN"

    #model = train_model_from_scratch(model_name)
    #load_and_train_model(os.path.join(OUTPUT_DIR, model_name, "model_saves", "model_epoch_1.hdf5"))
    #load_and_make_prediction(os.path.join(OUTPUT_DIR, model_name, "model_saves", "best_model_epoch_81.hdf5"), VALIDATION_SAMPLES_DIR)
    #load_and_make_prediction(os.path.join(OUTPUT_DIR, model_name, "model_saves", "model_epoch_143.hdf5"), VALIDATION_SAMPLES_DIR)
    #load_and_make_prediction_best_model(os.path.join(OUTPUT_DIR, model_name), VALIDATION_SAMPLES_DIR)

    def train_models(models_names):

        processes = list()
        paths = [os.path.join(OUTPUT_DIR, model_name, "model_saves", "model_epoch_1.hdf5") for model_name in models_names]
        
        for path in paths:
            p = Process(target=load_and_train_model, args=(path,))
            processes.append(p)
        
        for p in tqdm(processes, desc="Training models"):
            p.start()
            p.join()


    def predict_models(models_names):

        processes = list()
        paths = [os.path.join(OUTPUT_DIR, model_name) for model_name in models_names]
        
        for path in paths:
            p = Process(target=load_and_make_prediction_best_model, args=(path, VALIDATION_SAMPLES_DIR))
            processes.append(p)
        
        for p in tqdm(processes, desc="Predicting with models"):
            p.start()
            p.join()

    models_names = ["1e-5_lr_deep_high_filters_UnetCNN", #
                    "1e-1_lr_deep_high_filters_UnetCNN",#
                    "SGD_deep_high_filters_UnetCNN", #
                    "deep_high_high_filters_UnetCNN",#
                    "0.1_clipvalue_1e-4_lr_LeakyRelu_deep_high_filters_UnetCNN",#
                    "mae_deep_high_filters_UnetCNN",#
                    "1e-2_lr_deep_high_filters_UnetCNN",#
                    "ELU_deep_high_filters_UnetCNN",#
                    "RMSProp_deep_high_filters_UnetCNN", #
                    "faint_color_loss_deep_high_filters_UnetCNN",#
                    "BN_deep_high_filters_UnetCNN",#
                    "DROPOUT_deep_high_filters_UnetCNN",#
                    "DROPOUT_BN_deep_high_filters_UnetCNN",#
                    "1_perceptual_loss_faint_color_loss_deep_high_filters_UnetCNN",#
                    "1e-4_deep_high_filters_UnetCNN",
                    "weight_decay_0.9_AdamW_deep_high_filters_UnetCNN",
                    "BN_latent_high_filters_UNetCNN", 
                    "DROPOUT_latent_BN_latent_high_filters_UNetCNN",
                    "DROPOUT_latent_high_filters_UNetCNN",
                    "Naive_CNN",
                    "5ks_deep_high_filters_UnetCNN",
                    "conv_instead_of_maxpool_deep_high_filters_UnetCNN",
                    "deep_high_filters_ResUNetCNN", 
                    "encoder_res_deep_high_filters_UNetCNN",
                    "decoder_res_deep_high_filters_UNetCNN", 
                    "encoder_res_decoder_res_deep_high_filters_UNetCNN",
                    "AdaBelief_deep_high_filters_UnetCNN",
                    "Nadam_deep_high_filters_UnetCNN",
                    "Adamax_deep_high_filters_UnetCNN",
                    "Adadelta_deep_high_filters_UnetCNN", 
                    "Adagrad_deep_high_filters_UnetCNN",
                    "LazyAdam_deep_high_filters_UnetCNN",
                    "LAMB_deep_high_filters_UnetCNN",
                    ]


    #Test all models exist
    for model in models_names:
        assert model in os.listdir(OUTPUT_DIR), f"The model {model} is not present in the output directory"


    model_name = "1_perceptual_loss_faint_color_loss_deep_high_filters_UnetCNN"
    p = Process(target=load_and_make_prediction_best_model, args=(os.path.join(OUTPUT_DIR, model_name), VALIDATION_SAMPLES_DIR))
    p.start()
    p.join()

    print(f"\n\n###################Training {len(models_names[14:])} ycrcb models#######################\n\n")
    train_models(models_names[14:])
    print(f"\n\n###################Predicting {len(models_names)} ycrcb models#######################\n\n")
    predict_models(models_names)

if __name__ == "__main__":
    main()
