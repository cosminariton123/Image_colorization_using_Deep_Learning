from keras.models import Model

from data_loader import TrainingGenerator
from preprocessing import preprocess_image_training, preprocess_image_predicting

from paths import TRAIN_SAMPLES_DIR, VALIDATION_SAMPLES_DIR
from config import TRAINING_BATCH_SIZE, EPOCHS

from callbacks import generate_callbacks
from save_model_info import generate_print_and_log_summary, log_image_samples_at_the_start_of_training
from util import generate_log_dir_of_not_exists


def train_model_and_save(model: Model , save_path, initial_epoch=0):

    training_data_generator = TrainingGenerator(samples_dir=TRAIN_SAMPLES_DIR,
        batch_size=TRAINING_BATCH_SIZE,
        preprocessing_procedure=preprocess_image_training,
        shuffle=False
    )
    
    validation_data_generator = TrainingGenerator(
            samples_dir=VALIDATION_SAMPLES_DIR,
            batch_size=TRAINING_BATCH_SIZE,
            preprocessing_procedure=preprocess_image_predicting,
            shuffle=False
    )

    log_dir = generate_log_dir_of_not_exists(save_path)
    generate_print_and_log_summary(model, log_dir)
    log_image_samples_at_the_start_of_training(log_dir, initial_epoch, training_data_generator, validation_data_generator)

    model.fit(
        training_data_generator,
        epochs = EPOCHS,
        validation_data = validation_data_generator,
        callbacks = generate_callbacks(save_path),
        shuffle = False,
        initial_epoch = initial_epoch
    )
