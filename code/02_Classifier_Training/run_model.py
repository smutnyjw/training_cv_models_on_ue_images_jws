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
import json

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
#       Misc Method declarations
####################################################################

def setup_output_dir(base_path, type, case: str, cfg_num):
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
                        f"{start}-{type}-{case}-{cfg_num}")

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

def run_experiment(type: str, cfg_setting: int, CASE_FLAG: str, base_path: str):
    print(f"----- Run {type} model w/ cfg {cfg_setting} -----")
    ERROR_FLAG = False

    ########################
    # First determine what model is being tested
    if type.lower() == "vgg16":
        import cfg_vgg16 as cfg

        cfg.BASE_PATH = setup_output_dir(base_path=base_path,
                                         type='vgg16',
                                         case=CASE_FLAG,
                                         cfg_num=cfg_setting)

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
                                         type='mobilenet',
                                         case=CASE_FLAG,
                                         cfg_num=cfg_setting)

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
    # Load from settings json file

    with open(cfg.JSON_SETTINGS_FILE[CASE_FLAG]) as json_file:
        settings = json.load(json_file)

        init_epochs = settings[str(cfg_setting)]['init_epochs']
        total_epochs = settings[str(cfg_setting)]['total_epochs']
        total_synth_train = settings[str(cfg_setting)]['total_synth_training']
        total_real_train = settings[str(cfg_setting)]['total_real_training']
        total_real_val = settings[str(cfg_setting)]['total_real_val']
        total_real_test = settings[str(cfg_setting)]['total_real_test']
        total_real_images = total_real_train + total_real_val + total_real_test
        class_ratios = settings['class_ratios']


    ##########################
    # Override based on json file

    try:
        cfg.TOTAL_NUMBER_OF_IMAGES = total_real_images + total_synth_train

        cfg.TRAIN_PERC_REAL = total_real_images \
                              / (total_real_images + total_synth_train)

        cfg.NUM_INIT_EPOCHS = init_epochs
        cfg.EPOCHS = total_epochs

        cfg.PERC_TRAIN = (total_real_train + total_synth_train) \
                            / cfg.TOTAL_NUMBER_OF_IMAGES
        cfg.PERC_VAL = total_real_val / cfg.TOTAL_NUMBER_OF_IMAGES
        cfg.PERC_TEST = total_real_test / cfg.TOTAL_NUMBER_OF_IMAGES

        #[RealCat/Defect, RealDog/Good, SynCat/Defect, SynDog/Good]
        force_sample = [int(total_real_train * class_ratios[0]),
                        int(total_real_train * class_ratios[1]),
                        int(total_synth_train * class_ratios[0]),
                        int(total_synth_train * class_ratios[1])]
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
    if not math.isclose(cfg.PERC_TRAIN + cfg.PERC_VAL + cfg.PERC_TEST, 1.0):
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
    num_paths = 0
    if CASE_FLAG.lower() == "weld":
        '''
        Binary classifier to state whether a metal weld is a defect or not.
        '''

        # Checks - Do the data paths exist
        for key in cfg.DATASET_LISTS_WELD:
            for i, path in enumerate(cfg.DATASET_LISTS_WELD[key]):
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

        class_weights = {'Defect': 1,
                            'Good': 1
                     }

        cfg.BINARY_THRESHOLD = 0.5

        metric_to_monitor = 'val_auc'
        monitor_mode = "max"

        # Real Images
        dataset_real = cfg.DATASET_LISTS_WELD['real']

        # Synthetic images
        dataset_synth = cfg.DATASET_LISTS_WELD['synth']

    elif CASE_FLAG.lower() == "cats_dogs":
        '''
        Binary classifier to determine if the image is of a Cat or Dog
        '''

        # Checks - Do the data paths exist
        for key in cfg.DATASET_LISTS_CD:
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
        dataset_real = cfg.DATASET_LISTS_CD['real']

        # Synthetic images
        dataset_synth = cfg.DATASET_LISTS_CD['synth']
    else:
        raise("*** ERROR: Invalid CASE_FLAG string input. Must be either "
              "['weld', 'cats_dogs']. Terminating script.")

    # End of if-else statement to load data

    ##############################################################
    #               Sampling Synthetic Dataset(s)

    ##
    # Real data

    # 1) Load all real data
    num_loaded_real_imgs = 0
    population_real_df = pd.DataFrame()
    for id, dataset_path in enumerate(dataset_real):

        df_i = pd.read_csv(dataset_path, header=0, index_col=None)
        num_loaded_real_imgs += len(df_i)

        if id == 0:
            population_real_df = df_i.copy()
        else:
            population_real_df = pd.merge(population_real_df, df_i, how='outer')

    if len(population_real_df['label'].unique()) != 2:
        raise("ERROR: Number of unique classes found is not 2. Double check "
              "your loaded datasets.")

    # 3) Sample each class
    df_real = pd.DataFrame()
    for id, l in enumerate(population_real_df['label'].unique()):
        df_l = population_real_df.query(f'label == \"{l}\"')
        df_l = df_l.sample(int(total_real_images * class_ratios[id]))

        if id == 0:
            df_real = df_l.copy()
        else:
            df_real = pd.merge(df_real, df_l, how='outer')

    ##
    # Synth data

    # 1) Load all synth data
    num_loaded_synth_imgs = 0
    population_synth_df = pd.DataFrame()
    for id, dataset_path in enumerate(dataset_synth):

        df_i = pd.read_csv(dataset_path, header=0, index_col=None)
        num_loaded_synth_imgs += len(df_i)

        if id == 0:
            population_synth_df = df_i.copy()
        else:
            population_synth_df = pd.merge(population_synth_df,
                                           df_i, how='outer')

    if len(population_synth_df['label'].unique()) != 2:
        raise("ERROR: Number of unique classes found is not 2. Double check "
              "your loaded datasets.")

    # 3) Sample each class
    df_synth = pd.DataFrame()
    for id, l in enumerate(population_synth_df['label'].unique()):
        df_l = population_synth_df.query(f'label == \"{l}\"')
        df_l = df_l.sample(int(total_synth_train * class_ratios[id]))

        if id == 0:
            df_synth = df_l.copy()
        else:
            df_synth = pd.merge(df_synth, df_l, how='outer')


    # CHECK
    if len(df_real) + len(df_synth) > cfg.TOTAL_NUMBER_OF_IMAGES and \
            len(df_real) + len(df_synth) < cfg.TOTAL_NUMBER_OF_IMAGES * 0.98:
        ERROR_FLAG = True
        print("--- ERROR: Double check splitting of data. Unequal "
                        "number of images versus what is desired: "
                        f"{len(df_real) + len(df_synth)} vs "
                        f"{cfg.TOTAL_NUMBER_OF_IMAGES}")
    if list(df_real.keys()) != list(df_synth.keys()):
        ERROR_FLAG = True
        print("--- ERROR: The column keys for df_real and df_synth do not "
              f"match. real: {df_real.keys()} vs synth: {df_synth.keys()}. "
              "Please double check your data and match the column "
              "headers.")

    if ERROR_FLAG == True:
        raise Exception("--- ERROR: Please fix the above errors")

    ##############################################################
    #               Establish Train/Val/Test dataframes

    keys = df_real.keys()

    if int(total_real_train) > 0:
        x_train, x_val, y_train, y_val = train_test_split(
                            df_real[keys[0]].tolist(),
                            df_real[keys[1]].tolist(),
                            train_size=int(total_real_train),
                            stratify=df_real[keys[1]].tolist()
                        )

        x_val, x_test, y_val, y_test = train_test_split(
                            x_val,
                            y_val,
                            test_size=int(total_real_val),
                            stratify=y_val
                        )
    else:
        x_train, y_train = [], []

        x_val, x_test, y_val, y_test = train_test_split(
            df_real[keys[0]].tolist(),
            df_real[keys[1]].tolist(),
            test_size=int(total_real_val),
            stratify=df_real[keys[1]].tolist()
        )

    df_train_real = pd.DataFrame({keys[0]: x_train, keys[1]: y_train})

    df_train = pd.concat([df_train_real, df_synth])
    df_val = pd.DataFrame({keys[0]: x_val, keys[1]: y_val})
    df_test = pd.DataFrame({keys[0]: x_test, keys[1]: y_test})

    ##
    # Check
    df_master = pd.concat([df_train, df_val, df_test])
    for label in df_master[df_master.keys()[1]].unique():
        if label not in list(cfg.CLASS_MAP_.keys()):
            raise Exception(
                f"*** ERROR: Unknown class {label} found in df_master")

    ###
    # Output dataset artifacts
    labels = list(cfg.CLASS_MAP_.keys())

    print(f"df_train = {len(df_train[df_train['label'] == labels[0]])}/"
                f"{len(df_train[df_train['label'] == labels[1]])}")
    print(f"df_val = {len(df_val[df_val['label'] == labels[0]])}/"
                f"{len(df_val[df_val['label'] == labels[1]])}")
    print(f"df_test = {len(df_test[df_test['label'] == labels[0]])}/"
                f"{len(df_test[df_test['label'] == labels[1]])}")
    print("*************************\n")

    cfg._info_model["Classes"] = labels
    cfg._info_model["Number of REAL Images"] = total_real_images
    cfg._info_model["Number of REAL Training Images"] = total_real_train
    cfg._info_model["Number of SYNTH Images"] = total_synth_train

    cfg._info_model["Class Distribution"] = [
                                len(df_master[df_master['label'] == labels[0]]),
                                len(df_master[df_master['label'] == labels[1]])]
    cfg._info_model["Class Distribution - Real Images"] = [
                                len(df_real[df_real['label'] == labels[0]]),
                                len(df_real[df_real['label'] == labels[1]])]
    cfg._info_model["Class Distribution - Train Images"] = [
                                len(df_train[df_train['label'] == labels[0]]),
                                len(df_train[df_train['label'] == labels[1]])]
    cfg._info_model["Class Distribution - Val Images"] = [
                                len(df_val[df_val['label'] == labels[0]]),
                                len(df_val[df_val['label'] == labels[1]])]
    cfg._info_model["Class Distribution - Test Images"] = [
                                len(df_test[df_test['label'] == labels[0]]),
                                len(df_test[df_test['label'] == labels[1]])]

    # Update COST values based on class proportion in the real dataset
    num_class00 = len(df_master[df_master['label'] == labels[0]])
    num_class01 = len(df_master[df_master['label'] == labels[1]])
    if cfg_setting != 28:
        class_weights[labels[0]] = 1 - (num_class00 / len(df_master))
        class_weights[labels[1]] = 1 - (num_class01 / len(df_master))
    else:
        # Use for the No Change test in the ReSample Experiment
        class_weights[labels[0]] = 0.5
        class_weights[labels[1]] = 0.5
    cfg._2D_aug["Class Weights"] = [round(class_weights[labels[0]], 3),
                                    round(class_weights[labels[1]], 3)]

    cfg.COST_FP = class_weights[labels[1]]
    cfg.COST_FN = class_weights[labels[0]]
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

    for i, img in enumerate(list(df_train[df_train.columns[0]])):
        if os.path.isfile(img) is False:
            print(f"\tdf_train {i} file does not exist: {img}")

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
        f"*** DEBUG - Begin pre-training")

    hist_init = model.fit(
        train_generator,
        epochs=cfg.NUM_INIT_EPOCHS,
        validation_data=val_generator,
        class_weight={cfg.CLASS_MAP_[labels[0]]:class_weights[labels[0]],
                      cfg.CLASS_MAP_[labels[1]]:class_weights[labels[1]]}
    )

    print(f"*** DEBUG - After "
          f"{keras.backend.get_value(model.optimizer.iterations)} "
          f"pre-training iterations")

    hist = model.fit(
        train_generator,
        epochs=cfg.EPOCHS,
        initial_epoch=cfg.NUM_INIT_EPOCHS,
        validation_data=val_generator,
        class_weight={cfg.CLASS_MAP_[labels[0]]:class_weights[labels[0]],
                      cfg.CLASS_MAP_[labels[1]]:class_weights[labels[1]]},
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
    out.output_plots(base_path=cfg.BASE_PATH, history=hist,
                     train_imgs=[total_real_train, total_synth_train],
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
                                                      len(df_test_results), 5)

    out.output_debug_info(os.path.join(cfg.BASE_PATH, "_REF_run_info.txt"),
                          [cfg._info_model, cfg._info_dataset,
                           cfg._info_results])

    out.output_Confusion_Mat_Heatmap(
                conf_matx=conf_matx,
                path=os.path.join(cfg.BASE_PATH, "Test_Results_Heatmap.png"),
                train_imgs=[total_real_train, total_synth_train],
                cost=cost_per_test_sample,
                classes=list(df_test["label"].unique()))

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

    args = parser.parse_args()
    if args != 3:
        raise "ERROR: Incorrect number of inputs. Need two (-m & -c)."

    path = os.path.join('output')

    print("DONE")

    run_experiment(type=args.model,
                    cfg_setting=args.cfg,
                    CASE_FLAG=args.use_case,
                    base_path=path)


