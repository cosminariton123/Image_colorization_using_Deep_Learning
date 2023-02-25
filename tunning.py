import tensorflow as tf
from keras.models import Model

from data_loader import TrainingGenerator
from preprocessing import preprocess_image_training, preprocess_image_predicting

from paths import TRAIN_SAMPLES_DIR, VALIDATION_SAMPLES_DIR
from config import TRAINING_BATCH_SIZE, EPOCHS

from callbacks import generate_callbacks
from save_model_info import plot_history, generate_print_and_save_summary
from util import generate_log_dir_of_not_exists, log_image_samples_at_the_start_of_training


def train_model_and_save(model: Model , save_path, initial_epoch=0):

    generate_print_and_save_summary(model, save_path)

    training_data_generator = TrainingGenerator(samples_dir=TRAIN_SAMPLES_DIR,
        batch_size=TRAINING_BATCH_SIZE,
        preprocessing_procedure=preprocess_image_training,
        shuffle=True
    )
    
    validation_data_generator = TrainingGenerator(
            samples_dir=VALIDATION_SAMPLES_DIR,
            batch_size=TRAINING_BATCH_SIZE,
            preprocessing_procedure=preprocess_image_predicting,
            shuffle=True
    )

    log_dir = generate_log_dir_of_not_exists(save_path)

    log_image_samples_at_the_start_of_training(log_dir, training_data_generator, validation_data_generator)

    history = model.fit(
        training_data_generator,
        epochs = EPOCHS,
        validation_data = validation_data_generator,
        callbacks = generate_callbacks(save_path),
        shuffle = False,
        initial_epoch = initial_epoch
    )

    plot_history(history, save_path)
