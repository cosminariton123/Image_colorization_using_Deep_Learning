import cv2
from PIL import Image
import multiprocessing

#DATA CONFIG
INPUT_SIZE = (150, 150, 1)
GROUND_TRUTH_SIZE = (150, 150, 2)

#PREPROCESSING CONFIG
INTERPOLATION_RESIZE = cv2.INTER_CUBIC
INTERPOLATION_ROTATE = Image.Resampling.BICUBIC
NOISE_PERCENTAGE = 0.1
NOISE_MEAN = 0
NOISE_DEVIATION = 20
MAX_DEGREE_OF_LEFT_RIGHT_ROTATION = 30
MAX_BRIGHTNESS_AUGM_COEFICIENT = 0.2 #0 is unchanged, 1.2 is up to 20% more or less brightness

#PREPROCESSING INFERENCE CONFIG
RESIZE_TO_TRAINING_SIZE = True

#ML CONFIG
EPOCHS = 300
EARLY_STOPPING_PATIENTE_IN_EPOCHS = 81
EARLY_STOPPING_MIN_DELTA = 0
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 19
REDUCE_LR_COOLDOWN = 1
REDUCE_LR_MIN_DELTA = 0.000_01
REDUCE_LR_MIN_LR = 1e-6
TRAINING_BATCH_SIZE = 24 #Should be multiple of 8(even better 128 for TPUs) for better efficiency
PREDICTION_BATCH_SIZE = 1

    #Use this if you want to use your computer for something else
    #and performance of the pc is hindered by training
    #Fragmentation of memory will be higher
LIMIT_GPU_MEMORY_GROWTH = True
    #Use this if you want to have lower floating point precission(has almost no effect on loss),
    #but use less memory 
    #and compute faster for graphics cards with compute capability above 7
MIXED_PRECISION_16 = True

#OPTIMIZATIONS CONFIG
NR_OF_PROCESSES = multiprocessing.cpu_count()

#TENSORBOARD CONFIG
HISTOGRAM_FREQ = 1
WRITE_GRAPHS = True
WRITE_IMAGES = True
WRITE_STEPS_PER_SECOND = True
PROFILE_BATCH = (1, 100)
TRAINING_IMAGE_DIFFERENT_SAMPLES_TO_LOG = 3
TRAINING_IMAGE_SAME_SAMPLE_TO_LOG = 3
VALIDATION_IMAGE_SAMPLES_TO_LOG = 3

#MODEL SAVE CONFIG
SAVE_LAST_X_EPOCHS = 300
