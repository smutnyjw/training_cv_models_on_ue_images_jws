'''
File:   cfg_vgg16.py
Author: John Smutny
Date:   03/26/2024
Description:
    Settings file for the VGG16 model. This file defines items
    such as the TRAIN/VAL/TEST data split, learning rate, optimizer, binary
    classifier thresholds, etc.

    In addition, at the bottom of the file, debug output logs are defined for
    output at the end of each training run.
Other Notes:

'''

import os
import keras
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay

# Establish class keys IN VALUE ORDER
#   class: class_y_val
CLASS_MAP_ = {'dog': 0,
               'cat': 1}
NUM_CLASSES = len(list(CLASS_MAP_.keys()))

COST_TP = 0
COST_TN = 0
COST_FP = 10
COST_FN = 10
BINARY_THRESHOLD = 0.5

DATASET_LISTS_CD = {
    "real": [],
    "synth": [],
}

DATASET_LISTS_WELD = {
    "real": [],
    "synth": []
}

IMAGE_SIZE = (224, 224, 3)  # Any setting other than (224, 224, 3) is untested

TOTAL_NUMBER_OF_IMAGES = 10000
PERC_TRAIN = 0.80
PERC_VAL = 0.10
PERC_TEST = 0.10

TRAIN_PERC_REAL = 1.0
VAL_PERC_REAL = 1.0

EPOCHS = 100
NUM_INIT_EPOCHS = 5
BATCH_SIZE = 16
BATCH_SIZE_VAL = 8
BATCH_SIZE_TEST = BATCH_SIZE_VAL

MODEL_OPT = "SGD"
MODEL_LR = 0.001
MODEL_DECAY_STEPS = TOTAL_NUMBER_OF_IMAGES*PERC_TRAIN/BATCH_SIZE
MODEL_DECAY_RATE = 0.95


_2D_aug = {
    "HEADER": "Dataset Augmentation Information",
    "DatasetSplit": [PERC_TRAIN, PERC_VAL, PERC_TEST],
    "rotation_range": 15,
    'width_shift_range': 0.1,    # % of image
    'height_shift_range': 0.1,   # % of image
    'brightness_range': (0.1, 1.0),
    'shear_range': 10,
    'zoom_range': 0.15,          # [1-zoom_range, 1+zoom_range]
    'channel_shift': 30,
    'fill_mode': "nearest",     # 'constant', 'nearest', 'reflect', 'wrap'
    'horizontal_flip': True,
    'vertical_flip': True,
    "rescale": None,            # 1.0/255
    "data_format": "channels_last",  # (samples, height, width, channels)
    "BinaryThreshold": BINARY_THRESHOLD
}

_info_model = {
    "HEADER": "Model Information",
    "Model": "VGG16",
    "Classes": CLASS_MAP_.keys(),
    "Number of Images": TOTAL_NUMBER_OF_IMAGES,
    "Number of REAL Images": -1,
    "Number of SYNTH Images": -1,
    "Number of SYNTH Images available": -1,
    "Class Distribution": [-1, -1],
    "Class Distribution - Real Images": [-1, -1],
    "Class Distribution - Train Images": [-1, -1],
    "Class Distribution - Val Images": [-1, -1],
    "Class Distribution - Test Images": [-1, -1],
    "Class Weights": [1, 1],
    "Real_Synth ratio - TRAIN": [TRAIN_PERC_REAL, 1 - TRAIN_PERC_REAL],
    "Real_Synth ratio - VAL": [VAL_PERC_REAL, 1 - VAL_PERC_REAL],
    "Dataset Breakdown (TRAIN/VAL/TEST)": [PERC_TRAIN * TOTAL_NUMBER_OF_IMAGES,
                                           PERC_VAL * TOTAL_NUMBER_OF_IMAGES,
                                           PERC_TEST * TOTAL_NUMBER_OF_IMAGES],
    "Optimizer:": MODEL_OPT,
    "Learning Rate:": MODEL_LR,
    "Planned Epochs": EPOCHS,
    "Batch Sizes (TRAIN/VAL/TEST)": [BATCH_SIZE, BATCH_SIZE_VAL,
                                     BATCH_SIZE_TEST],
    "ImageSize": IMAGE_SIZE
}

_info_dataset = _2D_aug

_info_results = {
    "HEADER": "Run Results Information",
    "Total Epochs Trained": -1,
    "test_Manual_accuracy": -1,
    "test_cost_per_sample": -1,
    "test_Manual_TP": -1,
    "test_Manual_TN": -1,
    "test_Manual_FP": -1,
    "test_Manual_FN": -1,
    "test_accuracy": -1,
    "test_loss": -1,
    "test_recall": -1,
    "test_precision": -1,
    "test_auc": -1,
    "loss": -1,
    "accuracy": -1,
    "recall": -1,
    "precision": -1,
    "auc": -1,
    "val_loss": -1,
    "val_accuracy": -1,
    "val_recall": -1,
    "val_precision": -1,
    "Cost Values [TP/TN/FP/FN]": [COST_TP, COST_TN, COST_FP, COST_FN]
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




