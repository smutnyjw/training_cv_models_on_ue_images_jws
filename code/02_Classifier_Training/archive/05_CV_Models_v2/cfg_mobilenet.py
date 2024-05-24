'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import os
import random
from datetime import datetime
import keras
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay

# Establish class keys
#   class: class_y_val
CLASS_MAP_ = {'dog': 0,
               'cat': 1}
NUM_CLASSES = len(list(CLASS_MAP_.keys()))
BINARY_THRESHOLD = 0.5

DATASET = [
    os.path.join('..', '_data', 'abs_cats_dogs.csv')
]

IMAGE_SIZE = (224, 224, 3)

TOTAL_NUMBER_OF_IMAGES = 10000
PERC_TRAIN = 0.70
PERC_VAL = 0.10
PERC_TEST = 0.20

# Results log
#   Real    Split       Test_Acc
#   1.0     70/10/20    0.93
#   0.3     70/10/20    0.89
#   0.1     70/10/20    0.85

TRAIN_PERC_REAL = 1.0 #1.0= #0.3
VAL_PERC_REAL = 1.0

MODEL_OPT = "Adam"
MODEL_LR = 1.0e-4
lr_schedule = ExponentialDecay(
                                    initial_learning_rate=MODEL_LR,
                                    decay_steps=800,    # 3-5 epochs
                                    decay_rate=0.95,
                                    staircase=True)
MODEL_OPTIMIZER = keras.optimizers.Adam(learning_rate=lr_schedule)

EPOCHS = 100
BATCH_SIZE = 32
STEPS_PER_EPOCH_TRAIN = -1
STEPS_PER_EPOCH_VAL = -1

# Ideally: 1 epoch = whole image set
#   AKA - BATCH_SIZE * STEPS = Num_imgs

_2D_aug = {
    "HEADER": "Dataset Augmentation Information",
    "DatasetSplit": [PERC_TRAIN, PERC_VAL, PERC_TEST],
    "rotation_range": 10,
    'width_shift_range': 0,
    'height_shift_range': 0,
    'brightness_range': [0.0, 0.2],
    'shear_range': 0.2,
    'zoom_range': 0.2,          # [1-zoom_range, 1+zoom_range]
    'fill_mode': "nearest",     # 'constant', 'nearest', 'reflect', 'wrap'
    'horizontal_flip': True,
    'vertical_flip': True,
    "rescale": None,            # 1.0/255
    "data_format": "channels_last"  # (samples, height, width, channels)
}

_info_model = {
    "HEADER": "Model Information",
    "Model": "MobileNetV3",
    "Classes": CLASS_MAP_.keys(),
    "Number of Images": TOTAL_NUMBER_OF_IMAGES,
    "Number of REAL Images": -1,
    "Number of SYNTH Images": -1,
    "Number of SYNTH Images available": -1,
    "Real_Synth ratio - TRAIN": [TRAIN_PERC_REAL, 1-TRAIN_PERC_REAL],
    "Real_Synth ratio - VAL": [VAL_PERC_REAL, 1-VAL_PERC_REAL],
    "Dataset Breakdown (TRAIN/VAL/TEST)": [PERC_TRAIN * TOTAL_NUMBER_OF_IMAGES,
                                           PERC_VAL * TOTAL_NUMBER_OF_IMAGES,
                                           PERC_TEST * TOTAL_NUMBER_OF_IMAGES],
    "Optimizer:": MODEL_OPT,
    "Learning Rate:": MODEL_LR,
    "Planned Epochs": EPOCHS,
    "NumberOfEpochSteps": [0, 0],
    "ImageSize": IMAGE_SIZE,
}

_info_dataset = _2D_aug

_info_results = {
    "HEADER": "Run Results Information",
    "Total Epochs Trained": -1,
    "test_Manual_accuracy": -1,
    "test_Manual_TP_TN": -1,
    "test_accuracy": -1,
    "test_binary_accuracy": -1,
    "test_categorical_accuracy": -1,
    "test_loss": -1,
    "test_recall": -1,
    "test_precision": -1,
    "loss": -1,
    "accuracy": -1,
    "recall": -1,
    "precision": -1,
    "val_loss": -1,
    "val_accuracy": -1,
    "val_recall": -1,
    "val_precision": -1,
}

#########################################
# Artifact File paths
BASE_PATH = os.path.join('output')

#########################################
# Configuration checks

if PERC_TRAIN + PERC_VAL + PERC_TEST != 1.0:
    sum = PERC_TRAIN + PERC_VAL + PERC_TEST
    PERC_TRAIN = PERC_TRAIN / sum
    PERC_VAL = PERC_VAL / sum
    PERC_TEST = PERC_TEST / sum

    print("*** WARN: Dataset percentage is not ==1. Setting dataset "
          f"percentages as {PERC_TRAIN}/{PERC_VAL}/{PERC_TEST}")

if "HEADER" not in _info_model.keys() or \
        "HEADER" not in _info_dataset.keys() or \
        "HEADER" not in _info_results.keys():
    raise Exception("*** ERROR: Debug dictionaries need a 'HEADER' key.")

if NUM_CLASSES != len(list(CLASS_MAP_.keys())):
    raise Exception("*** ERROR: Difference between declared NUM_CLASSES and "
                    "the defined CLASS_MAP_")

for i in range(len(CLASS_MAP_.values())):
    if i == 0:
        prev_value = list(CLASS_MAP_.values())[0]
    else:
        if list(CLASS_MAP_.values())[i] < prev_value:
            raise Exception("*** ERROR: Must have CLASS_MAP_ in value order")

        prev_value = list(CLASS_MAP_.values())[i]
