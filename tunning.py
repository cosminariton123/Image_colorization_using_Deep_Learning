import os

import tensorflow as tf
from keras.models import Model

from data_loader import TrainingGenerator
from paths import TRAIN_SAMPLES_DIR, VALIDATION_SAMPLES_DIR
from preprocessing import preprocess_image_training, preprocess_image_predicting
from save_model_info import plot_history, save_summary
from util import remove_old_models_until_x_most_recent_remain, remove_old_best_models

from config import TRAINING_BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENTE_IN_EPOCHS, EARLY_STOPPING_MIN_DELTA
from config import REDUCE_LR_COOLDOWN, REDUCE_LR_PATIENCE, REDUCE_LR_MIN_DELTA, REDUCE_LR_MIN_LR, REDUCE_LR_FACTOR
from config import HISTOGRAM_FREQ, WRITE_GRAPHS, WRITE_IMAGES, WRITE_STEPS_PER_SECOND, PROFILE_BATCH
from config import SAVE_LAST_X_EPOCHS

def generate_summary(model: Model):
    summary = []
    model.summary(print_fn=lambda line: summary.append(line))
    summary = "\n".join(summary)
    
    return summary


def generate_callbacks(save_path):
    callbacks = list()
    
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())

    callbacks.append(tf.keras.callbacks.EarlyStopping(
        min_delta = EARLY_STOPPING_MIN_DELTA,
        patience = EARLY_STOPPING_PATIENTE_IN_EPOCHS,
        verbose = 1
    ))

    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        factor = REDUCE_LR_FACTOR,
        patience = REDUCE_LR_PATIENCE,
        verbose = 1,
        min_delta = REDUCE_LR_MIN_DELTA,
        cooldown = REDUCE_LR_COOLDOWN,
        min_lr = REDUCE_LR_MIN_LR,
        mode="min"
    ))

    model_saves_dir = os.path.join(save_path, "model_saves")
    if not os.path.exists(model_saves_dir):
        os.mkdir(model_saves_dir)

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(model_saves_dir ,"best_model_epoch_{epoch}.hdf5"),
    verbose = 1,
    save_best_only = True,
    mode="min"
    ))

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(model_saves_dir, "model_epoch_{epoch}.hdf5"),
    verbose = 1,
    mode="min"
    ))

    callbacks.append(tf.keras.callbacks.LambdaCallback(
        on_epoch_end = lambda epoch, logs: remove_old_best_models(epoch, model_saves_dir)
    ))

    callbacks.append(tf.keras.callbacks.LambdaCallback(
        on_epoch_end = lambda epoch, logs: remove_old_models_until_x_most_recent_remain(epoch, model_saves_dir, SAVE_LAST_X_EPOCHS)
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


def train_model_and_save(model: Model , save_path, initial_epoch=0):

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
        
        initial_epoch = initial_epoch
    )

    plot_history(history, save_path)
