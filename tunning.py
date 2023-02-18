import os

import tensorflow as tf
from keras.models import Sequential

from data_loader import TrainingGenerator
from paths import TRAIN_SAMPLES_DIR, VALIDATION_SAMPLES_DIR
from preprocessing import preprocess_image_training, preprocess_image_predicting
from save_model_info import plot_history, save_summary
from config import TRAINING_BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENTE_IN_EPOCHS, HISTOGRAM_FREQ, WRITE_GRAPHS, WRITE_IMAGES, WRITE_STEPS_PER_SECOND, PROFILE_BATCH

def generate_summary(model: Sequential):
    summary = []
    model.summary(show_trainable=True, print_fn=lambda line: summary.append(line))
    summary = "\n".join(summary)
    
    return summary


def generate_callbacks(save_path):
    callbacks = list()

    callbacks.append(tf.keras.callbacks.EarlyStopping(
        patience=EARLY_STOPPING_PATIENTE_IN_EPOCHS,
    ))

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                                    filepath = os.path.join(save_path ,"best_model.hdf5"),
                                    save_only_best_model = True
                                    ))

    log_dir = os.path.join(save_path, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir = log_dir,
        histogram_freq = HISTOGRAM_FREQ,
        write_graph = WRITE_GRAPHS,
        write_images = WRITE_IMAGES,
        write_steps_per_second = WRITE_STEPS_PER_SECOND,
        profile_batch = PROFILE_BATCH
    ))

    return callbacks


def train_model_and_save(model: Sequential , save_path):

    summary = generate_summary(model)
    save_summary(summary, save_path)
    print(summary)

    callbacks = generate_callbacks(save_path)

    history = model.fit(
        TrainingGenerator(samples_dir=TRAIN_SAMPLES_DIR,
        batch_size=TRAINING_BATCH_SIZE,
        preprocessing_procedure=preprocess_image_training,
        shuffle=True
        ),

        epochs = EPOCHS,

        validation_data = TrainingGenerator(
            samples_dir=VALIDATION_SAMPLES_DIR,
            batch_size=TRAINING_BATCH_SIZE,
            preprocessing_procedure=preprocess_image_predicting,
            shuffle=True
            ),

        callbacks = [callbacks],

        shuffle = False,
    )

    plot_history(history, save_path)
