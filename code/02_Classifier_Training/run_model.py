'''
File:   run_model.py
Author: John Smutny
Date:   03/26/2024
Description: 
    This python file contains all logic and steps to test two transfer
    learning models {vgg16, mobilenet} for binary classification in two use
    cases {cats_dogs, weld}.

    The file 1) loads datasets of real and synthetically generated images,
    2) builds a particular model, 3) defines Training/Validation/Test
    datasets, 4) trains and tests the model, then 5) outputs various graphs,
    charts, and info files for future reference.

Other Notes:
    Execution occurs in the ::main() function.

    WARN - Keras.ImageDataGenerator() is debricated.
        It is suggested that the user convert to other methods
        such as the 'tf.keras.utils.image_dataset_from_directory and transforming
        via tf.data.Dataset preprocessing layers.
        https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

    WARN - Must use keras version 2.13.1 or earlier

'''
import math
import os, gc
import numpy as np

from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime

import tensorflow as tf
import keras
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

import lib_output_artifacts as out
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


####################################################################
####################################################################
#                   Model Declarations
####################################################################
####################################################################


def build_mobilenet_transfer_learning(img_size, num_classes, base_path):
    '''
    Construct a transfer learning model from the published MobileNetV3-small
    model published by Andrew Howard, et al. in the paper
    "Searching for mobilenetv3.
    :param img_size: [height, width, ch] Dimensions of processed images
    :param num_classes: How many classes the classifier should consider (2 only)
    :param base_path: Absolute file path to the current run's output directory
    :return: NA

    Creates artifacts
        - Base MobileNetV3 model architecture.
        - Full Transfer Learning model architecture.
    '''
    num_out_nodes = 1
    act_fct = "sigmoid"

    # Load base MobileNetV3
    # https://keras.io/api/applications/mobilenet/
    mobilenet_model = keras.applications.MobileNetV3Small(
        input_shape=img_size,       #default is (224,224,3)
        alpha=1.0,    # default=1    Causes error if not 1
        minimalistic=False,
        include_top=False,
        weights="imagenet"
    )

    # User input checks
    if num_classes != 2:
        print(f"ERROR - Declared an unsupported number of classes {num_classes}")
        os._exit()

    ###
    # Construct the paper's model architecture with minor modifications.
    input = keras.layers.Input(shape=img_size)
    base_model_layers = mobilenet_model(input)

    # b1_pool = keras.layers.GlobalAveragePooling2D(
    #     data_format="channels_last",
    #     keepdims=True)(base_model_layers)

    b1_pool = keras.layers.AveragePooling2D(
                        (7, 7),
                        data_format="channels_last",
                        name='pool_7x7')(base_model_layers)

    # Replication of the paper's output layers using h-swish residual layers.
    #   h-swish = [b1] * [b2] = [x] * [ReLU(x+3) / 6]
    b1_conv = keras.layers.Conv2D(filters=512,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        use_bias=False,
                        kernel_initializer=keras.initializers.GlorotNormal(),
                        kernel_regularizer=keras.regularizers.L2(l2=1e-4),
                        # kernel_regularizer=keras.regularizers.L2(l2=1e-5),
                        name='Conv_2')(b1_pool)

    # Start of branching residual model
    b2_lambda_add_h = keras.layers.Lambda( lambda x: tf.math.add(x, 3),
                            #lambda x: x + 3,
                            name="h_swish-tf_TFOpLambda_add_3")(b1_conv)
    b2_relu_h = keras.layers.ReLU(name='h_swish-re_lu')(b2_lambda_add_h)
    b2_multiply_h = keras.layers.Lambda(
                        lambda x: tf.math.multiply(x, 1/6),
                        name="h_swish-tf_TFOpLambda_multiply_6th")(b2_relu_h)

    # Close of residual model branches
    b3_multiply_h = keras.layers.Multiply()([b1_conv, b2_multiply_h])
    b3_dropout = keras.layers.Dropout(0.8)(b3_multiply_h)
    b3_logits = keras.layers.Conv2D(filters=256,
                        kernel_size=(1, 1),
                        kernel_initializer=keras.initializers.GlorotNormal(),
                        kernel_regularizer=keras.regularizers.L2(l2=1e-4),
                        # kernel_regularizer=keras.regularizers.L2(l2=1e-5),
                        name='logits') (b3_dropout)

    b3_flatten = keras.layers.Flatten()(b3_logits)
    output = keras.layers.Dense(num_out_nodes,
                                activation=act_fct)(b3_flatten)

    res_model = keras.models.Model(input, output)

    ###
    # Set layers of mobilenet to be trainable to yield 30-50% params trainable
    # -9 = ~33% - 407,000
    # -16 = ~46% - 560,000 - expanded_conv_10 (Conv2D)
    # -32 = ~66% - 700,000 - expanded_conv_9/project (Conv2)
    num_nontrain_layers = len(res_model.layers[1].layers) - 9 #16 #32 #9
    for l in range(num_nontrain_layers):
        res_model.layers[1].layers[l].trainable = False

    ####
    # OUTPUT model artifacts
    path_mobilenet = os.path.join(base_path,
                                f"mobilenet_{num_classes}_trainable_layers.txt")
    out.output_model_arch(path_mobilenet, mobilenet_model)
    path_model = os.path.join(base_path,
                                f"model_{num_classes}_architecture.txt")
    out.output_model_arch(path_model, res_model)

    return res_model


def build_vgg16_transfer_learning(img_size, num_classes, base_path):
    '''
    Construct a transfer learning model from the published VGG16
    model published by Karen Simonyan and Andrew Zisserman in the paper:
    "Very deep convolutional networks for large-scale image recognition"
    :param img_size: [height, width, ch] Dimensions of processed images
    :param num_classes: How many classes the classifier should consider (2 only)
    :param base_path: Absolute file path to the current run's output directory
    :return: NA

    Creates artifacts
        - Base VGG16 model architecture.
        - Full Transfer Learning model architecture.
    '''
    num_out_nodes = 1
    act_fct = "sigmoid"

    # Load base VGG16
    vgg16_model = keras.applications.VGG16(
        input_shape=img_size,
        include_top=False,
        weights="imagenet"
    )

    if num_classes != 2:
        print(
            f"ERROR - Declared an unsupported number of classes {num_classes}")
        os._exit()

    ###
    # Construct the paper's model architecture with minor modifications.
    model = keras.models.Sequential([
        # The base VGG16 model
        vgg16_model,

        # here is our custom prediction layer
        keras.layers.Flatten(),
        keras.layers.Dense(256,
                           activation='relu',
                           kernel_initializer=keras.initializers.GlorotNormal(),
                           kernel_regularizer=keras.regularizers.L2(l2=5e-4),
                           name="fc1"),
        keras.layers.Dense(256,
                           activation='relu',
                           kernel_initializer=keras.initializers.GlorotNormal(),
                           kernel_regularizer=keras.regularizers.L2(l2=5e-4),
                           name="fc2"),
        keras.layers.Dense(num_out_nodes, activation=act_fct,
                           name="Prediction")
    ])

    ###
    # Set x layers of vgg16 to be trainable
    num_nontrain_layers = len(vgg16_model.layers) - 2
    for l in range(num_nontrain_layers):
        vgg16_model.layers[l].trainable = False

    ###
    # OUTPUT model artifacts
    path_vgg16 = os.path.join(base_path,
                                f"vgg16_{num_classes}_trainable_layers.txt")
    out.output_model_arch(path_vgg16, vgg16_model)
    path_model = os.path.join(base_path,
                                f"model_{num_classes}_architecture.txt")
    out.output_model_arch(path_model, model)

    return model

####################################################################
#       Experiment Setting
####################################################################

def run_a_specific_setting(cfg_setting):
    '''
    This function is set by the user to enable running a particular test
    multiple times sequentially but with different settings each time. This
    increases test efficiency.
    :param cfg_setting:
    :return: [number of epochs to train,
                Total amount of images to use,
                Percentage of images used in the TRAINING set are Real]
    '''
    epochs = -1
    total_images = -1
    train_perc_real = -1
    num_init_epochs = -1
    force_sample = [0, 0, 0, 0]

    print(f"*** WARN: cfg settings override {cfg_setting}")
    if cfg_setting == 0:
        total_images = 10000
        train_perc_real = 1.0
    elif cfg_setting == 1:
        total_images = 10000
        train_perc_real = 0.75
    elif cfg_setting == 2:
        total_images = 10000
        train_perc_real = 0.50
    elif cfg_setting == 3:
        total_images = 10000
        train_perc_real = 0.25
    elif cfg_setting == 4:
        total_images = 10000
        train_perc_real = 0.10
    elif cfg_setting == 5:
        total_images = 10000
        train_perc_real = 0.05
    elif cfg_setting == 6:
        total_images = 10000
        train_perc_real = 0.0
    elif cfg_setting == 11:  # 75:25 --> 6000 train real images
        total_images = 7500
        train_perc_real = 1.0
    elif cfg_setting == 12:  # 50:50 --> 4000 train real images
        total_images = 5000
        train_perc_real = 1.0
    elif cfg_setting == 13:     # 25:75 --> 2000 train real images
        total_images = 2500
        train_perc_real = 1.0
    elif cfg_setting == 14:     # 10:90 --> 800 train real images
        total_images = 1000
        train_perc_real = 1.0
    elif cfg_setting == 15:     # 05:95 --> 400 train real images
        total_images = 500
        train_perc_real = 1.0

    elif cfg_setting == 20:  # -->  RSR - no change
        total_images = 2000
        train_perc_real = 1.0
        force_sample = [1400, 600, 0, 0]    #[RealDef,RealGo,SynDef,SynGo]
    elif cfg_setting == 21:  # -->  Resample - downsample
        total_images = 1200
        train_perc_real = 1.0
        force_sample = [600, 600, 0, 0]
    elif cfg_setting == 22:  # -->  Resample - upsample
        total_images = 2400
        train_perc_real = 1.0
        force_sample = [1200, 1200, 0, 0]
    elif cfg_setting == 23:  # -->  Resample - pad w/ synth
        total_images = 2400
        train_perc_real = 0.6875
        force_sample = [1200, 600, 0, 600]
    elif cfg_setting == 24:  # -->  RSR
        total_images = 2500
        train_perc_real = 0.75
        force_sample = [1400, 600, 250, 250]
    elif cfg_setting == 25:  # -->  RSR
        total_images = 3000
        train_perc_real = 0.58
        force_sample = [1400, 600, 580, 420]
    elif cfg_setting == 26:  # -->  RSR
        total_images = 4000
        train_perc_real = 0.37
        force_sample = [1400, 600, 1240, 760]
    elif cfg_setting == 27:  # -->  RSR
        total_images = 5000
        train_perc_real = 0.25
        force_sample = [1400, 600, 2100, 900]
    elif cfg_setting == 28:  # -->  Resample - No Change, No Class Weighting
        total_images = 2000
        train_perc_real = 1.0
        force_sample = [1200, 600, 0, 0]  # [RealDef,RealGo,SynDef,SynGo]
    elif cfg_setting == 29:  # -->  Resample - No Change, Class Weighting
        total_images = 2000
        train_perc_real = 1.0
        force_sample = [1200, 600, 0, 0]  # [RealDef,RealGo,SynDef,SynGo]

    else:
        num_init_epochs = 3
        if cfg_setting == -1:
            epochs = 10
            total_images = 2000
            train_perc_real = 1.0
        if cfg_setting == -2:
            epochs = 10
            total_images = 2000
            train_perc_real = 0.9
        elif cfg_setting == -3:
            epochs = 10
            total_images = 2000
            train_perc_real = 0.25
        elif cfg_setting == -4:
            epochs = 10
            total_images = 2000
            train_perc_real = 0.1
        elif cfg_setting == -5:
            epochs = 10
            total_images = 2000
            train_perc_real = 0.05
        elif cfg_setting == -6:
            epochs = 10
            total_images = 2000
            train_perc_real = 0.0
        elif cfg_setting == -99:
            epochs = 1
            total_images = 1500
            train_perc_real = 1.0
        else:
            print(f"WARN - Running from .cfg files. Invalid cfg_setting"
                  f" {cfg_setting}")

    return [epochs, total_images, train_perc_real,
            num_init_epochs, force_sample]

####################################################################
#       Misc Method declarations
####################################################################

def setup_output_dir(base_path, type, cfg_num):
    '''
    Function used at the start of any run to create a 'run artifact'
    directory. This directory will contain various output files and
    information from the training run.
    :param base_path:
    :param type: Which model is being trained {vgg16, mobilenet}
    :param cfg_num: A numeric run id. The number is only for reference.
    :return:
    '''
    now = datetime.now()
    start = now.strftime("%Y_%m_%d-%H_%M_%S")
    path = os.path.join(base_path, type, 'pending_review',
                        f"{start}-{type}-{cfg_num}")

    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

    return path


def make_binary_prediction(model, thres, x_array):
    confidence = model.predict(x_array, verbose=0)    # No msg
    prediction = 0 if confidence < thres else 1
    return prediction, confidence


def process_test_image(img, show_image):
    if show_image:
        print(img)
        plt.imshow(img)

    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x

####################################################################
#           Dataset creation Method declarations
####################################################################

def validate_dataset_size(df_data_size: int,
                          desired_num_imgs: int):
    if desired_num_imgs > df_data_size:
        raise Exception(
            "*** ERROR: You have requested too much data for this "
            f"dataset collection. "
            f"Request: {desired_num_imgs}, Dataset: {df_data_size}")
    else:
        num_images = desired_num_imgs

    return int(num_images)

# TODO - Edit to take in a dictionary of {'class':NumSamples} to sample by
#  class.
def binary_sample_dataset(df_data: pd.DataFrame, desired_samples):
    keys = df_data.keys()
    x = df_data[keys[0]].tolist()
    y = df_data[keys[1]].tolist()

    if desired_samples > 0:
        _, x, _, y = train_test_split(x, y,
                                      test_size=desired_samples,
                                      stratify=y)
        df = pd.DataFrame({df_data.keys()[0]: x, df_data.keys()[1]: y})
    else:
        print("*** WARN - ::binary_sample_dataset(): "
              "Number of requested samples from the loaded dataframe is zero. "
              "Returning empty Dataframe")
        df = pd.DataFrame(columns=df_data.keys())

    return df


def assign_data_splits(df_real: pd.DataFrame, df_synth: pd.DataFrame,
                       split_perc: list, real_synth_ratio: list):
    ####
    # Split the loaded data into TRAINING, VALIDATION, and TEST sets

    total_images = len(df_real) + len(df_synth)
    keys = df_real.keys()

    # Shuffle real data that will be sub-divided into train/val/test
    df_real.sample(1)

    # TRAIN_REAL - Only take the training samples that is needed
    if real_synth_ratio[0] > 0.0:
        num_needed_train_real = int(total_images *
                                    split_perc[0] * real_synth_ratio[0]
                                    # split_perc[1] * real_synth_ratio[1]
                                    )
        x_train, x_val, y_train, y_val = train_test_split(
            df_real[keys[0]].tolist(),
            df_real[keys[1]].tolist(),
            train_size=num_needed_train_real,
            stratify=df_real[keys[1]].tolist()
        )

        # VAL_REAL & TEST_REAL - Take all remaining samples for VAL & TEST sets
        val_test_split = split_perc[2] / (split_perc[2] + split_perc[1])
        x_val, x_test, y_val, y_test = train_test_split(
            x_val,
            y_val,
            test_size=val_test_split,
            stratify=y_val
        )

        # Create resulting datasets
        df_train = pd.concat([df_synth,
                              pd.DataFrame(
                                  {keys[0]: x_train,
                                   keys[1]: y_train})])
    else:
        # VAL_REAL & TEST_REAL - Take all remaining samples for VAL & TEST sets
        val_test_split = split_perc[2] / (split_perc[2] + split_perc[1])
        x_val, x_test, y_val, y_test = train_test_split(
                            df_real[keys[0]].tolist(),
                            df_real[keys[1]].tolist(),
                            test_size=val_test_split,
                            stratify=df_real[keys[1]].tolist()
        )

        # Since PERC_TRAIN_REAL == 0.0, then training data is all synth
        x_train = []
        df_train = df_synth
    # End of if-else statement

    # Assign VAL & TEST data to all REAL data
    df_val = pd.DataFrame({keys[0]: x_val, keys[1]: y_val})
    df_test = pd.DataFrame({keys[0]: x_test, keys[1]: y_test})

    # Print debug info
    total_real = len(x_train) + len(df_val) + len(df_test)
    total_synth = len(df_synth)
    print(
        f"*** DEBUG: Train Test Split: {len(x_train)}_{len(df_synth)}/"
        f"{len(df_val)}/{len(df_test)} == "
        f"{len(x_train) + len(df_synth) + len(df_val) + len(df_test)}")
    print(f"*** DEBUG: Real {total_real} vs Synth {total_synth} count")

    # Shuffle all dataframes
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_val = df_val.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    return df_train, df_val, df_test, [total_real, total_synth]
    # End of ::assign_data_splits() function


####################################################################
####################################################################
#                           MAIN
####################################################################
####################################################################

def run_experiment(type: str, cfg_setting: int, CASE_FLAG: str,
                    base_path: str, use_synth_dataset_list: list):
    print(f"----- Run {type} model w/ cfg {cfg_setting} -----")

    ########################
    # First determine what model is being tested
    if type.lower() == "vgg16":
        import cfg_vgg16 as cfg

        cfg.BASE_PATH = setup_output_dir(base_path=base_path,
                                         type='vgg16', cfg_num=cfg_setting)
        model = build_vgg16_transfer_learning(img_size=cfg.IMAGE_SIZE,
                                                num_classes=cfg.NUM_CLASSES,
                                                base_path=cfg.BASE_PATH,
                                              )
        lr_schedule = ExponentialDecay(
            initial_learning_rate=cfg.MODEL_LR,
            decay_steps=cfg.MODEL_DECAY_STEPS,
            decay_rate=cfg.MODEL_DECAY_RATE,
            staircase=True)

        optimizer = keras.optimizers.SGD(learning_rate=lr_schedule,
                                               momentum=0.9)

    elif type.lower() == "mobilenetv3":
        import cfg_mobilenet as cfg
        cfg.BASE_PATH = setup_output_dir(base_path=base_path,
                                         type='mobilenet', cfg_num=cfg_setting)

        model = build_mobilenet_transfer_learning(img_size=cfg.IMAGE_SIZE,
                                                num_classes=cfg.NUM_CLASSES,
                                                base_path=cfg.BASE_PATH)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=cfg.MODEL_LR,
            decay_steps=cfg.MODEL_DECAY_STEPS,
            decay_rate=cfg.MODEL_DECAY_RATE,
            staircase=True)

        optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule,
                                               momentum=0.9)

    else:
        raise Exception(f"--- ERROR: Invalid model selection ({type}). "
                        f"Chose either [vgg16, mobilenetv3].")

    ##########################
    # Override for specific settings
    override_settings = run_a_specific_setting(cfg_setting)
    try:
        if override_settings[0] != -1:
            cfg.EPOCHS = override_settings[0]
        if override_settings[1] != -1:
            cfg.TOTAL_NUMBER_OF_IMAGES = override_settings[1]
        if override_settings[2] != -1:
            cfg.TRAIN_PERC_REAL = override_settings[2]
        if override_settings[3] != -1:
            cfg.NUM_INIT_EPOCHS = override_settings[3]
        force_sample = override_settings[4]
    except:
        raise("*** ERROR after ::run_a_specific_setting() - could not parse "
              "one of the outputs. Please run code in 'debug' mode.")

    # Fill in debug run information.
    cfg._info_model["Number of Images"] = cfg.TOTAL_NUMBER_OF_IMAGES
    cfg._info_model["Planned Epochs"] = cfg.EPOCHS
    cfg._info_model["DatasetSplit"] = [cfg.PERC_TRAIN,
                                       cfg.PERC_VAL, cfg.PERC_TEST]

    cfg._info_model["Real_Synth ratio - TRAIN"] = [cfg.TRAIN_PERC_REAL,
                                                    1 - cfg.TRAIN_PERC_REAL]
    cfg._info_model[ "Real_Synth ratio - VAL"] = [cfg.VAL_PERC_REAL,
                                                    1 - cfg.VAL_PERC_REAL]
    cfg._info_model["Dataset Breakdown (TRAIN/VAL/TEST)"] = \
                                [cfg.PERC_TRAIN * cfg.TOTAL_NUMBER_OF_IMAGES,
                                cfg.PERC_VAL * cfg.TOTAL_NUMBER_OF_IMAGES,
                                cfg.PERC_TEST * cfg.TOTAL_NUMBER_OF_IMAGES]

    # Checks
    if cfg.PERC_TRAIN + cfg.PERC_VAL + cfg.PERC_TEST != 1.0:
        raise Exception("--- ERROR: Dataset splits must equal 1.0")
    if cfg.TRAIN_PERC_REAL < 0.0 or cfg.TRAIN_PERC_REAL > 1.0:
        raise Exception("--- ERROR: Percentage of data that is real for "
                        "training split must be b/w 0.0 & 1.0")

    ############################################
    ############################################
    #                   BEGIN
    ############################################
    ############################################

    # WARN - Keras.ImageDataGenerator() is debricated. Convert to other methods
    #  such as the 'tf.keras.utils.image_dataset_from_directory and transforming
    #  via tf.data.Dataset preprocessing layers.
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

    # Must use keras version 2.13.1 or earlier

    ############################################
    ############################################

    # Establish debugging .txt. This will be appended throughout the run.
    out.output_debug_info(os.path.join(cfg.BASE_PATH, "_REF_run_info.txt"),
                          [cfg._info_model, cfg._info_dataset,
                           cfg._info_results])

    # Establish all ImageDataGenerator classes to augment Training data and
    # take in validation+test data
    train_datagen = ImageDataGenerator(
        rotation_range=cfg._2D_aug['rotation_range'],
        width_shift_range=cfg._2D_aug['width_shift_range'],
        height_shift_range=cfg._2D_aug['height_shift_range'],
        brightness_range=cfg._2D_aug['brightness_range'],
        shear_range=cfg._2D_aug['shear_range'],
        zoom_range=cfg._2D_aug['zoom_range'],
        channel_shift_range=cfg._2D_aug['channel_shift'],
        fill_mode=cfg._2D_aug['fill_mode'],
        horizontal_flip=cfg._2D_aug['horizontal_flip'],
        vertical_flip=cfg._2D_aug['vertical_flip'],
        rescale=cfg._2D_aug['rescale'],
        data_format=cfg._2D_aug['data_format'],
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    ####################################################################
    ####################################################################
    # Load specific datasets for the specific use case
    if CASE_FLAG.lower() == "weld":
        '''
        Binary classifier to state whether a metal weld is a defect or not.
        '''

        # Checks
        for key in cfg.DATASET_LISTS_CD:
            for i, path in enumerate(cfg.DATASET_LISTS_CD[key]):
                if os.path.exists(path) is False:
                    raise Exception(
                        f"--- ERROR: The following DATASET_LISTS_CD "
                        f"path does not exist: {path}")
                num_paths += 1
        if num_paths == 0:
            raise Exception("--- ERROR: Zero dataset paths listed. Please "
                            "specify a dataset .csv file to load.")


        # TODO - Confirm Heatmap & class_weight information is correct.


        # Setup the various classfier variables for this case
        cfg.CLASS_MAP_ = {'Defect': 0,
                            'Good': 1,
                        }

        class_weights = {'Defect': -1,
                            'Good': -1
                     }

        cfg.BINARY_THRESHOLD = 0.5

        metric_to_monitor = 'val_auc'
        monitor_mode = "max"

        # Real Images
        abs_path_real = cfg.DATASET_LISTS_WELD['real']

        # Synthetic images
        abs_path_synths = cfg.DATASET_LISTS_WELD['synth']

    elif CASE_FLAG.lower() == "cats_dogs":
        '''
        Binary classifier to determine if the image is of a Cat or Dog
        '''

        # Checks
        for key in cfg.DATASET_LISTS_WELD:
            for i, path in enumerate(cfg.DATASET_LISTS_CD[key]):
                if os.path.exists(path) is False:
                    raise Exception(f"--- ERROR: The following "
                                    f"DATASET_LISTS_CD path does not exist:"
                                    f" {path}")
                num_paths += 1
        if num_paths == 0:
            raise Exception("--- ERROR: Zero dataset paths listed. Please "
                            "specify a dataset .csv file to load.")

        # Setup the various classfier variables for this case
        cfg.CLASS_MAP_ = {'dog': 0,
                            'cat': 1}

        class_weights = {'dog': 1,
                            'cat': 1}

        cfg.BINARY_THRESHOLD = 0.5

        metric_to_monitor = 'val_loss'
        monitor_mode = "min"

        # Real Images
        abs_path_real = cfg.DATASET_LISTS_CD['real']

        # Synthetic images
        abs_path_synths = cfg.DATASET_LISTS_CD['synth']
    else:
        raise("*** ERROR: Invalid CASE_FLAG string input. Must be either "
              "['weld', 'cats_dogs']. Terminating script.")

    # End of if-else statement to load data

    # Check
    if len(abs_path_synths) != len(use_synth_dataset_list):
        print("*** WARN: User specified a 'user_synth_dataset_list' "
              "register that is not equal to the number of synthetic "
              "datasets available ({len(use_synth_dataset_list)}). "
              "Please correct your input and enter as a "
              "{len(abs_path_synths)}-bit register.")

    ##############################################################
    #               Sampling Synthetic Dataset(s)

    if sum(force_sample) > 0:
        # Select data as specified for each class and each quantity

        ##
        # Assumption:
        #   index = [#RealDefects, #RealGood, #SynthDefects, #SynthGood]
        ##

        df_real_dataset = pd.read_csv(abs_path_real, header=0, index_col=None)

        df_real_defect = df_real_dataset[df_real_dataset['class'] ==
                                       'Defect'].sample(force_sample[0])

        # Oversample if needed for 'GOOD' class
        if force_sample[1] >= \
            len(df_real_dataset[df_real_dataset['class'] == 'Good']):
            more_good_needed = force_sample[1] - \
                               len(df_real_dataset[df_real_dataset['class'] == 'Good'])

            df_real_good = df_real_dataset[df_real_dataset['class'] == 'Good']

            df_real_good = pd.concat([df_real_good,
                                      df_real_good.sample(more_good_needed)])

        else:
            df_real_good = df_real_dataset[df_real_dataset['class'] ==
                                            'Good'].sample(force_sample[1])

        df_real = pd.concat([df_real_defect, df_real_good])

        if force_sample[2] + force_sample[3] > 0:
            abs_syn_file = abs_path_synths[0]
            df_synth_dataset = pd.read_csv(abs_syn_file, header=0, index_col=None)

            if force_sample[2] > 0:
                df_synth_defect = df_synth_dataset[df_synth_dataset['class'] ==
                                             'Defect'].sample(force_sample[2])
            if force_sample[3] > 0:
                df_synth_good = df_synth_dataset[df_synth_dataset['class'] ==
                                           'Good'].sample(force_sample[3])

            if force_sample[2] > 0 and force_sample[3] > 0:
                df_synth = pd.concat([df_synth_defect, df_synth_good])
            elif force_sample[2] > 0:
                df_synth = df_synth_defect
            elif force_sample[3] > 0:
                df_synth = df_synth_good

            print(f"df_real = {len(df_real[df_real['class'] == 'Defect'])}/"
                  f"{len(df_real[df_real['class'] == 'Good'])}")
            print(f"df_synth = {len(df_synth[df_synth['class'] == 'Defect'])}/"
                  f"{len(df_synth[df_synth['class'] == 'Good'])}")
        else:
            df_synth = pd.DataFrame()

        # TODO - WORKAROUND - 'weld' case can only sample from the first
        #  synthetic dataset listed.
        #  Need to join the 'binary_sampling' for both cases.
        total_synth_images = len(df_synth)

    else:
        # Determine the proportion of data is from each loaded dataset
        total_synth_images = 0
        synth_dataframes_used = []
        sampling_ratio = []
        for i in range(len(use_synth_dataset_list)):

            if use_synth_dataset_list[i] and i < len(abs_path_synths):
                abs_path_synth = abs_path_synths[i]
                df_synth = pd.read_csv(abs_path_synth, header=0, index_col=None)
                total_synth_images += len(df_synth)

                synth_dataframes_used.append(df_synth)

                sampling_ratio.append(len(df_synth))
            elif i >= len(abs_path_synths):
                print("*** WARN: Attempting to select invalid synthetic dataset "
                      "#{i+1}. Only {len(abs_path_synths)} datasets are specified")

        for j in range(len(sampling_ratio)):
            sampling_ratio[j] = sampling_ratio[j] / total_synth_images

        ####
        # Sample the loaded synthetic data proportionally to the amount the
        #       datasets contribute
        desired_num_synth = int(cfg.TOTAL_NUMBER_OF_IMAGES * \
                                cfg.PERC_TRAIN * (1 - cfg.TRAIN_PERC_REAL))
        validate_dataset_size(total_synth_images, desired_num_synth)

        for i in range(len(synth_dataframes_used)):
            df = synth_dataframes_used[i]
            num_each_set = int(sampling_ratio[i] * desired_num_synth)
            if i == 0:
                df_synth = binary_sample_dataset(df, desired_samples=num_each_set)
            else:
                df_sy = binary_sample_dataset(df, desired_samples=num_each_set)

                df_synth = pd.concat([df_synth, df_sy])

        ##############################################################
        #                   Sampling Real Dataset(s)

        # Determine how many pictures are needed for the REAL portion
        df_real_dataset = pd.read_csv(abs_path_real, header=0, index_col=None)
        perc_train_real = cfg.PERC_TRAIN * cfg.TRAIN_PERC_REAL
        desired_num_real = int(cfg.TOTAL_NUMBER_OF_IMAGES * (perc_train_real +
                                                             cfg.PERC_VAL +
                                                             cfg.PERC_TEST))
        num_real = validate_dataset_size(len(df_real_dataset),
                                         desired_num_real)
        df_real = binary_sample_dataset(df_real_dataset, num_real)


    # CHECK
    if len(df_real) + len(df_synth) > cfg.TOTAL_NUMBER_OF_IMAGES and \
            len(df_real) + len(df_synth) < cfg.TOTAL_NUMBER_OF_IMAGES * 0.98:
        raise Exception("--- ERROR: Double check splitting of data. Unequal "
                        "number of images versus what is desired: "
                        f"{len(df_real) + len(df_synth)} vs "
                        f"{cfg.TOTAL_NUMBER_OF_IMAGES}")

    ##############################################################
    #               Combine sampled Real & Synthetic Data

    # Combine all of the sampled data into one Master file
    df_master = pd.concat([df_real, df_synth])

    ###
    # Assign Train/Val/Test splits
    # TODO - This method is sampling with a non-even rate



    if 20 <= cfg_setting <= 30:
        # Shuffle real data that will be sub-divided into train/val/test
        df_real.sample(1)

        if len(df_synth) > 0:
            df_synth.sample(1)

        # Determine slice size
        num_train = int(len(df_master) * cfg.PERC_TRAIN)
        num_val = int(len(df_master) * cfg.PERC_VAL)
        num_test = int(len(df_master) * cfg.PERC_TEST)

        # Assign Validation data
        val_ids = range(int(num_val/2))
        df_val = pd.concat([df_real[df_real['class']=='Defect'].iloc[val_ids],
                            df_real[df_real['class']=='Good'].iloc[val_ids]])

        test_ids = range(int(num_val/2), int(num_val/2+num_test/2))
        df_test = pd.concat([df_real[df_real['class']=='Defect'].iloc[test_ids],
                            df_real[df_real['class']=='Good'].iloc[test_ids]])

        d_ids = range(int(num_val/2+num_test/2),
                          len(df_real[df_real['class']=='Defect']))
        g_ids = range(int(num_val/2+num_test/2),
                          len(df_real[df_real['class']=='Good']))

        df_train = pd.concat([df_real[df_real['class']=='Defect'].iloc[d_ids],
                                df_real[df_real['class']=='Good'].iloc[g_ids],
                                df_synth])

        num_real_in_dataset = len(df_real)
        num_synth_in_dataset = len(df_synth)

    else:
        dataset_splits = assign_data_splits(df_real=df_real,
                                            df_synth=df_synth,
                                            split_perc=[cfg.PERC_TRAIN,
                                                        cfg.PERC_VAL,
                                                        cfg.PERC_TEST],
                                            real_synth_ratio=[cfg.TRAIN_PERC_REAL,
                                                              cfg.VAL_PERC_REAL]
                                            )
        # Decode Train/Val/Test splits
        df_train = dataset_splits[0]
        df_val = dataset_splits[1]
        df_test = dataset_splits[2]

        num_real_in_dataset = dataset_splits[3][0]
        num_synth_in_dataset = dataset_splits[3][1]

    # Should I have the test set be balanced?
    print(f"df_train = {len(df_train[df_train['class'] == 'Defect'])}/"
                f"{len(df_train[df_train['class'] == 'Good'])}")
    print(f"df_val = {len(df_val[df_val['class'] == 'Defect'])}/"
                f"{len(df_val[df_val['class'] == 'Good'])}")
    print(f"df_test = {len(df_test[df_test['class'] == 'Defect'])}/"
                f"{len(df_test[df_test['class'] == 'Good'])}")
    print("*************************\n")

    ##
    # Check
    for label in df_master[df_master.keys()[1]].unique():
        if label not in list(cfg.CLASS_MAP_.keys()):
            raise Exception(
                f"*** ERROR: Unknown class {label} found in df_master")

    ###
    # Output dataset artifacts
    keys = list(cfg.CLASS_MAP_.keys())
    cfg._info_model["Classes"] = keys
    cfg._info_model["Number of SYNTH Images available"] = total_synth_images
    cfg._info_model["Number of REAL Images"] = num_real_in_dataset
    cfg._info_model["Number of SYNTH Images"] = num_synth_in_dataset

    cfg._info_model["Class Distribution"] = [
                                len(df_master[df_master['class'] == keys[0]]),
                                len(df_master[df_master['class'] == keys[1]])]
    cfg._info_model["Class Distribution - Real Images"] = [
                                len(df_real[df_real['class'] == keys[0]]),
                                len(df_real[df_real['class'] == keys[1]])]
    cfg._info_model["Class Distribution - Train Images"] = [
                                len(df_train[df_train['class'] == keys[0]]),
                                len(df_train[df_train['class'] == keys[1]])]
    cfg._info_model["Class Distribution - Val Images"] = [
                                len(df_val[df_val['class'] == keys[0]]),
                                len(df_val[df_val['class'] == keys[1]])]
    cfg._info_model["Class Distribution - Test Images"] = [
                                len(df_test[df_test['class'] == keys[0]]),
                                len(df_test[df_test['class'] == keys[1]])]

    # Update COST values based on class proportion in the real dataset
    num_class00 = len(df_master[df_master['class'] == keys[0]])
    num_class01 = len(df_master[df_master['class'] == keys[1]])
    if cfg_setting != 28:
        class_weights[keys[0]] = 1 - (num_class00 / len(df_master))
        class_weights[keys[1]] = 1 - (num_class01 / len(df_master))
    else:
        # Use for the No Change test in the ReSample Experiment
        class_weights[keys[0]] = 0.5
        class_weights[keys[1]] = 0.5
    cfg._2D_aug["Class Weights"] = [round(class_weights[keys[0]], 3),
                                    round(class_weights[keys[1]], 3)]

    cfg.COST_FP = class_weights[keys[1]]
    cfg.COST_FN = class_weights[keys[0]]
    cfg._info_results["Cost Values [TP/TN/FP/FN]"] = [cfg.COST_TP, cfg.COST_TN,
                                                      cfg.COST_FP, cfg.COST_FN]

    out.output_dataset_csv(
        path=os.path.join(cfg.BASE_PATH, "img_list_TRAIN.csv"),
        df=df_train)
    out.output_dataset_csv(path=os.path.join(cfg.BASE_PATH, "img_list_VAL.csv"),
                           df=df_val)
    out.output_dataset_csv(
        path=os.path.join(cfg.BASE_PATH, "img_list_TEST.csv"),
        df=df_test)

    # End of if-else statement

    ####################################################################
    ####################################################################
    # Insert loaded image into training generator to apply 2D augmentations

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=None,
        x_col=df_train.columns[0],
        y_col=list(df_train.columns[1:])[0],
        classes=list(cfg.CLASS_MAP_.keys()),
        target_size=cfg.IMAGE_SIZE[:2],
        batch_size=cfg.BATCH_SIZE,
        # save_to_dir="..\\output\\aug_images\\train",
        # save_format='png',
        class_mode='binary'
    )

    #
    # NOTE: 2D Augmentations are only applied to TRAINING data.
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        directory=None,
        x_col=df_val.columns[0],
        y_col=list(df_val.columns[1:])[0],
        classes=list(cfg.CLASS_MAP_.keys()),
        target_size=cfg.IMAGE_SIZE[:2],
        batch_size=cfg.BATCH_SIZE_VAL,
        # save_to_dir="..\\output\\aug_images\\val",
        # save_format='png'
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test,
        directory=None,
        x_col=df_test.columns[0],
        y_col=list(df_test.columns[1:])[0],
        classes=list(cfg.CLASS_MAP_.keys()),
        target_size=cfg.IMAGE_SIZE[:2],
        batch_size=cfg.BATCH_SIZE_TEST,
        # save_to_dir="..\\output\\aug_images\\test",
        # save_format='png',
        class_mode='binary'
    )

    out.output_debug_info(os.path.join(cfg.BASE_PATH, "_REF_run_info.txt"),
                          [cfg._info_model, cfg._info_dataset,
                           cfg._info_results])

    #####################################################################
    #                        Model Training
    #####################################################################

    print("*** DEBUG: to train 100% of data, steps_per_epoch for current "
          f"data must be: \n\t- TRAIN {train_generator.n / cfg.BATCH_SIZE}"
          f"\n\t- VAL {val_generator.n / cfg.BATCH_SIZE_VAL}")

    checkpoint = ModelCheckpoint(os.path.join(cfg.BASE_PATH,
                                              f"basic_impl_{type}.h5"),
                                 monitor=metric_to_monitor,
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='auto',
                                 save_freq='epoch')

    patience = 7
    early = EarlyStopping(monitor=metric_to_monitor,
                          mode=monitor_mode,
                          min_delta=0.001,
                          patience=patience,
                          restore_best_weights=True,
                          # start_from_epoch=3, - Cannot use since tf=2.10,
                          # need tf>=2.11 but Windows Native GPU support ends
                          # at tf==2.10
                          verbose=1)

    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  #loss_weights=,
                  metrics=['accuracy',
                           keras.metrics.Recall(thresholds=None),  # TP+FN
                           keras.metrics.Precision(thresholds=None),  # TP+FP
                           keras.metrics.AUC(num_thresholds=100,
                                 curve="ROC",
                                 #label_weights=list(class_weights.values()),
                                 from_logits=True)
                           ]
                  )

    print(
        f"*** DEBUG - before pre-training LR = {keras.backend.get_value(model.optimizer.iterations)}")

    hist_init = model.fit(
        train_generator,
        epochs=cfg.NUM_INIT_EPOCHS,
        validation_data=val_generator,
        class_weight={cfg.CLASS_MAP_[keys[0]]:class_weights[keys[0]],
                      cfg.CLASS_MAP_[keys[1]]:class_weights[keys[1]]}
    )

    print(f"*** DEBUG - After pre-training LR = {keras.backend.get_value(model.optimizer.iterations)}")

    hist = model.fit(
        train_generator,
        epochs=cfg.EPOCHS+cfg.NUM_INIT_EPOCHS,
        initial_epoch=cfg.NUM_INIT_EPOCHS,
        validation_data=val_generator,
        class_weight={cfg.CLASS_MAP_[keys[0]]:class_weights[keys[0]],
                      cfg.CLASS_MAP_[keys[1]]:class_weights[keys[1]]},
        #callbacks=[checkpoint, early]
        callbacks=[early]
    )

    print(
        f"*** DEBUG - After training LR = {keras.backend.get_value(model.optimizer.iterations)}")

    try:
        for key in hist_init.history.keys():
            hist.history[key] = hist_init.history[key] + hist.history[key]
    except:
        print("ERROR - missing key to append histories.")

    ####
    # Output artifacts from training
    rs_ratio = f"{math.ceil(cfg.TRAIN_PERC_REAL*100)}:" \
               f"{math.ceil((1-cfg.TRAIN_PERC_REAL)*100)}"

    out.output_plots(base_path=cfg.BASE_PATH, history=hist, rs_ratio=rs_ratio,
                     num_init_epochs=cfg.NUM_INIT_EPOCHS)

    if monitor_mode == 'max':
        best_epoch = hist.history[metric_to_monitor].index(
                        max(hist.history[metric_to_monitor][cfg.NUM_INIT_EPOCHS:]))
    else:
        best_epoch = hist.history[metric_to_monitor].index(
                        min(hist.history[metric_to_monitor][cfg.NUM_INIT_EPOCHS:]))

    cfg._info_results['Total Epochs Trained'] = best_epoch + 1
    try:
        for key in hist.history.keys():
            cfg._info_results[key] = hist.history[key][best_epoch]
    except:
        print("ERROR - Tried to input into struct _info_results a key that "
              "didn't exist."
              f"\nhist = {hist.history}\n_info_results = {cfg._info_results}")

    out.output_debug_info(os.path.join(cfg.BASE_PATH, "_REF_run_info.txt"),
                          [cfg._info_model, cfg._info_dataset,
                           cfg._info_results])


    #####################################################################
    #                           TESTING
    #####################################################################
    # Test the fully trained model

    print("\n***** Model Testing *****")
    # Test the model against the test data
    scores = model.evaluate(test_generator)
    try:
        cfg._info_results['test_accuracy'] = scores[1]
        cfg._info_results['test_loss'] = scores[0]
        cfg._info_results['test_recall'] = scores[2]
        cfg._info_results['test_precision'] = scores[3]
        cfg._info_results['test_auc'] = scores[4]
    except IndexError:
        print("*** WARN: index error when accessing a member of 'score' after "
              f"testing. The size of 'score' returned in {len(scores)}")
        cfg._info_results['test_accuracy'] = scores
        out.output_debug_info(os.path.join(cfg.BASE_PATH, "_REF_run_info.txt"),
                              [cfg._info_model, cfg._info_dataset,
                               cfg._info_results])

    print(f"--- General Test Score: "
          f"Loss={round(scores[0], 3)}\tAccuracy={round(scores[1], 3)}")

    print(f"--- Manual Testing START: ")
    df_test_results = pd.DataFrame(columns=['filepath',
                                            'pred', 'actual', 'result',
                                            'confidence'])

    # Perform the manual test by manually calling .predict()
    for i in range(len(df_test)):
        if i % int(len(df_test) / 10) == 0:
            print(f"\t\t\t{int(i / len(df_test) * 100)}% complete...")
        img_path = df_test.iloc()[i, 0]
        label = df_test.iloc()[i, 1]

        img = keras.utils.load_img(img_path, target_size=cfg.IMAGE_SIZE)
        img = process_test_image(img, False)
        prediction, confidence = make_binary_prediction(model,
                                                  cfg.BINARY_THRESHOLD,
                                                  img)

        # Decode the prediction
        result = -1
        for key in cfg.CLASS_MAP_.keys():
            if cfg.CLASS_MAP_[key] == prediction:
                prediction = key
                result = 1 if prediction == label else 0
                break

        # store results in a dataframe
        df_test_results.loc[len(df_test_results.index)] = [img_path,
                                                           prediction,
                                                           label,
                                                           result,
                                                           confidence[0][0]]
    out.output_dataset_csv(path=os.path.join(cfg.BASE_PATH,
                                             "img_list_TEST_RESULTS.csv"),
                           df=df_test_results)

    # Analyze the manual test results
    conf_matx = confusion_matrix(df_test_results['actual'],
                                 df_test_results['pred'])

    TP, FP, FN, TN = conf_matx.ravel()
    cfg._info_results['test_Manual_TP'] = round(TP / len(df_test_results), 3) * 100
    cfg._info_results['test_Manual_TN'] = round(TN / len(df_test_results), 3) * 100
    cfg._info_results['test_Manual_FP'] = round(FP / len(df_test_results), 3) * 100
    cfg._info_results['test_Manual_FN'] = round(FN / len(df_test_results), 3) * 100

    cost = TP*cfg.COST_TP + TN*cfg.COST_TN + FP*cfg.COST_FP + FN*cfg.COST_FN
    cost_per_test_sample =  round(cost / len(df_test_results), 3)
    cfg._info_results['test_cost_per_sample'] = cost_per_test_sample

    num_correct = TP + TN
    cfg._info_results['test_Manual_accuracy'] = round(num_correct /
                                                      len(df_test_results), 2)

    out.output_debug_info(os.path.join(cfg.BASE_PATH, "_REF_run_info.txt"),
                          [cfg._info_model, cfg._info_dataset,
                           cfg._info_results])

    out.output_Confusion_Mat_Heatmap(
                conf_matx=conf_matx,
                path=os.path.join(cfg.BASE_PATH, "Test_Results_Heatmap.png"),
                rs_ratio=rs_ratio,
                cost=cost_per_test_sample,
                classes=list(df_test["class"].unique()))

    print(f"--- Manual Test Score: "
          f"Accuracy={round(num_correct / len(df_test_results), 3)} "
          f"= ({num_correct})")

    print("***** Testing Complete")


    ####################################################################
    #       Garbage Collection
    ####################################################################

    # (For multi-script execution) Reset keras backend session to release variables
    # reset Keras Session - https://github.com/keras-team/keras/issues/12625
    def reset_keras(classifier):
        sess = tf.compat.v1.keras.backend.get_session()
        tf.compat.v1.keras.backend.clear_session()
        sess.close()
        sess = tf.compat.v1.keras.backend.get_session()

        try:
            del classifier
        except:
            pass

        # use the same config as you used to create the session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.visible_device_list = "0"
        tf.compat.v1.keras.backend.set_session(
            tf.compat.v1.Session(config=config))

    reset_keras(model)

    # Force delete various variables in the scope of the python kernel
    del model, train_generator, val_generator, test_generator, checkpoint, \
        hist, scores, early

    # Prompt the pc Garbage Collector to free up unused memory.
    gc.collect()


if __name__=="__main__":
    ################
    # This version of run_model could in theory be used to run ::main() from the
    # command line in a powershell terminal.
    #################
    import argparse

    print("*** RUN __main__")

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", required=True, type=str,
                        help="[VGG16, MobileNetV3] model to run")
    parser.add_argument("-c", "--cfg", required=True, type=int,
                        help="[-1, 7] The set configuration to run")
    parser.add_argument("-u", "--use_case", required=True, type=str,
                        help="[Cats_Dogs, weld] The use case to run.")
    parser.add_argument("-s", "--synth_dataset_preset", required=True, type=int,
                        help="[0, 1] Choose which dataset presets to use")

    args = parser.parse_args()
    if args != 4:
        raise "ERROR: Incorrect number of inputs. Need two (-m & -c)."

    if args.synth_datasets == 0:
        synth_datasets = [1, 1, 0, 0]
    else:
        synth_datasets = [1, 1, 1, 1]


    path = os.path.join('output')

    print("DONE")

    main(type=args.model,
            cfg_setting=args.cfg,
            CASE_FLAG=args.use_case,
            base_path=path,
            use_synth_dataset_list=synth_datasets)


