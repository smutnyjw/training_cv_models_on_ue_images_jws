'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from random import randint
from datetime import datetime

import keras
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

import lib_output_artifacts as out

##################################
#       Model Declarations
##################################

def setup_output_dir(base_path, type, cfg_num):
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
    pred = model.predict(x_array, verbose=0)    # No msg
    prediction = 0 if pred < thres else 1
    #print('Predicted: ', pred)
    return prediction, pred

def make_multiclass_prediction(model, x_array):
    pred = model.predict(x_array, verbose=0)    # No msg
    prediction = np.argmax(pred, axis=-1)[0]
    #print('Predicted: ', pred)
    return prediction

def build_mobilenet_transfer_learning(img_size, num_classes, optimizer,
                                      base_path):
    # Load base MobileNetV3

    # https://keras.io/api/applications/mobilenet/
    mobilenet_model = keras.applications.MobileNetV3Small(
        input_shape=img_size,       #default is (224,224,3)
        alpha=1.0,    # default=1    Causes error
        minimalistic=False,
        include_top=False,
        weights="imagenet",
        pooling="avg",
        #classes=,
        #dropout_rate=,
        classifier_activation=None  #"softmax"
    )

    # TODO - Confirm operation b/w 2-classes and 3+-classes
    if num_classes == 2:
        num_out_nodes = 1
        act_fct = "sigmoid"
        loss_fct = "binary_crossentropy"
    elif num_classes > 2:
        num_out_nodes = num_classes
        act_fct = "softmax"
        loss_fct = "categorical_crossentropy"

    model = keras.models.Sequential([
        # our mobilenet model added as a layer
        mobilenet_model,
        # here is our custom prediction layer
        #   - https://keras.io/api/layers/initializers/
        #   - https://keras.io/api/layers/regularizers/
        keras.layers.Flatten(),
        keras.layers.Dropout(0.50),
        keras.layers.Dense(1024,
                           activation='relu',
                           kernel_initializer=keras.initializers.GlorotNormal(),
                           kernel_regularizer=keras.regularizers.L2(l2=0.001)),
        keras.layers.Dropout(0.20),
        keras.layers.Dense(512,
                           activation='relu',
                           kernel_initializer=keras.initializers.GlorotNormal(),
                           kernel_regularizer=keras.regularizers.L2(l2=0.001)),
        keras.layers.Dropout(0.10),
        keras.layers.Dense(num_out_nodes, activation=act_fct)
    ])

    # Set x layers of mobilenet to be trainable
    num_nontrain_layers = len(mobilenet_model.layers) - 24
    for l in range(num_nontrain_layers):
        mobilenet_model.layers[l].trainable = False

    # mobilenet_model.trainable = False
    keras.backend.clear_session()
    model.compile(optimizer=optimizer,
                  loss=loss_fct,
                  metrics=['accuracy', 'binary_accuracy',
                           keras.metrics.Recall(thresholds=None),  # TP+FN
                           keras.metrics.Precision(thresholds=None)  # TP+FP
                           ]
                  )

    # OUTPUT artifacts
    path_mobilenet = os.path.join(base_path,
                                f"mobilenet_{num_classes}_trainable_layers.txt")
    out.output_model_arch(path_mobilenet, mobilenet_model)
    path_model = os.path.join(base_path,
                                f"model_{num_classes}_architecture.txt")
    out.output_model_arch(path_model, model)

    return model

def process_test_image(img, show_image):
    if show_image:
        print(img)
        plt.imshow(img)

    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x

def build_vgg16_transfer_learning(img_size, num_classes, optimizer, base_path):
    # Load base VGG16
    vgg16_model = keras.applications.VGG16(
        input_shape=img_size,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    # TODO - Confirm operation b/w 2-classes and 3+-classes
    if num_classes == 2:
        num_out_nodes = 1
        act_fct = "sigmoid"
        loss_fct = "binary_crossentropy"
    elif num_classes > 2:
        num_out_nodes = num_classes
        act_fct = "softmax"
        loss_fct = "categorical_crossentropy"

    model = keras.models.Sequential([
        # our vgg16_base model added as a layer
        vgg16_model,
        # here is our custom prediction layer
        keras.layers.Flatten(),
        keras.layers.Dropout(0.50),
        keras.layers.Dense(1024,
                           activation='relu',
                           kernel_initializer=keras.initializers.GlorotNormal(),
                           kernel_regularizer=keras.regularizers.L2(l2=0.001)),
        keras.layers.Dropout(0.20),
        keras.layers.Dense(512,
                           activation='relu',
                           kernel_initializer=keras.initializers.GlorotNormal(),
                           kernel_regularizer=keras.regularizers.L2(l2=0.001)),
        keras.layers.Dropout(0.10),
        keras.layers.Dense(num_out_nodes, activation=act_fct)
    ])

    # Set x layers of vgg16 to be trainable
    num_nontrain_layers = len(vgg16_model.layers) - 4
    for l in range(num_nontrain_layers):
        vgg16_model.layers[l].trainable = False

    # vgg16_model.trainable = False
    keras.backend.clear_session()
    model.compile(optimizer=optimizer,
                  loss=loss_fct,
                  metrics=['accuracy', 'binary_accuracy',
                           keras.metrics.Recall(thresholds=None),  # TP+FN
                           keras.metrics.Precision(thresholds=None)  # TP+FP
                           ]
                  )

    # OUTPUT artifacts
    path_vgg16 = os.path.join(base_path,
                                f"vgg16_{num_classes}_trainable_layers.txt")
    out.output_model_arch(path_vgg16, vgg16_model)
    path_model = os.path.join(base_path,
                                f"model_{num_classes}_architecture.txt")
    out.output_model_arch(path_model, model)

    return model


##################################

def main(type: str, cfg_setting: int, CASE_FLAG: str, base_path: str):
    print(f"----- Run {type} model w/ cfg {cfg_setting} -----")

    ########################
    # First determine what model is being tested
    if type.lower() == "vgg16":
        import cfg_vgg16 as cfg
        model_optimizer = cfg.MODEL_OPTIMIZER
        cfg.BASE_PATH = setup_output_dir(base_path=base_path,
                                         type='vgg16', cfg_num=cfg_setting)
        model = build_vgg16_transfer_learning(img_size=cfg.IMAGE_SIZE,
                                                num_classes=cfg.NUM_CLASSES,
                                                optimizer=model_optimizer,
                                                base_path=cfg.BASE_PATH,
                                              )
    elif type.lower() == "mobilenetv3":
        import cfg_mobilenet as cfg
        model_optimizer = cfg.MODEL_OPTIMIZER
        cfg.BASE_PATH = setup_output_dir(base_path=base_path,
                                         type='mobilenet', cfg_num=cfg_setting)
        model = build_mobilenet_transfer_learning(img_size=cfg.IMAGE_SIZE,
                                                num_classes=cfg.NUM_CLASSES,
                                                optimizer=model_optimizer,
                                                base_path=cfg.BASE_PATH)
    else:
        raise Exception(f"--- ERROR: Invalid model selection ({type}). "
                        f"Chose either [vgg16, mobilenetv3].")


    ##########################
    # Override for specific settings
    if cfg_setting == 0:
        cfg.EPOCHS = 100
        cfg.TOTAL_NUMBER_OF_IMAGES = 10000
        cfg.PERC_TRAIN = 0.7
        cfg.PERC_VAL = 0.1
        cfg.PERC_TEST = 0.2
        cfg.TRAIN_PERC_REAL = 1.0
    else:
        print(f"*** WARN: cfg settings override {cfg_setting}")
        if cfg_setting == -1:
            cfg.EPOCHS = 1
            cfg.TOTAL_NUMBER_OF_IMAGES = 600
            cfg.PERC_TRAIN = 0.7
            cfg.PERC_VAL = 0.1
            cfg.PERC_TEST = 0.2
            cfg.TRAIN_PERC_REAL = 1.0
        if cfg_setting == 99:
            cfg.EPOCHS = 100
            cfg.TOTAL_NUMBER_OF_IMAGES = 10000
            cfg.PERC_TRAIN = 0.7
            cfg.PERC_VAL = 0.1
            cfg.PERC_TEST = 0.2
            cfg.TRAIN_PERC_REAL = round(randint(0, 100) / 100, 2)
        elif cfg_setting == 1:
            cfg.EPOCHS = 100
            cfg.TOTAL_NUMBER_OF_IMAGES = 10000
            cfg.PERC_TRAIN = 0.7
            cfg.PERC_VAL = 0.1
            cfg.PERC_TEST = 0.2
            cfg.TRAIN_PERC_REAL = 0.3
        elif cfg_setting == 2:
            cfg.EPOCHS = 100
            cfg.TOTAL_NUMBER_OF_IMAGES = 10000
            cfg.PERC_TRAIN = 0.7
            cfg.PERC_VAL = 0.1
            cfg.PERC_TEST = 0.2
            cfg.TRAIN_PERC_REAL = 0.10
        elif cfg_setting == 3:
            cfg.EPOCHS = 100
            cfg.TOTAL_NUMBER_OF_IMAGES = 10000
            cfg.PERC_TRAIN = 0.85
            cfg.PERC_VAL = 0.05
            cfg.PERC_TEST = 0.10
            cfg.TRAIN_PERC_REAL = 0.1
        elif cfg_setting == 4:
            cfg.EPOCHS = 100
            cfg.TOTAL_NUMBER_OF_IMAGES = 10000
            cfg.PERC_TRAIN = 0.85
            cfg.PERC_VAL = 0.05
            cfg.PERC_TEST = 0.10
            cfg.TRAIN_PERC_REAL = 0.05
        elif cfg_setting == 5:
            cfg.EPOCHS = 100
            cfg.TOTAL_NUMBER_OF_IMAGES = 10000
            cfg.PERC_TRAIN = 0.91
            cfg.PERC_VAL = 0.04
            cfg.PERC_TEST = 0.05
            cfg.TRAIN_PERC_REAL = 0.011
        elif cfg_setting == 6:
            cfg.EPOCHS = 100
            cfg.TOTAL_NUMBER_OF_IMAGES = 10000
            cfg.PERC_TRAIN = 0.90
            cfg.PERC_VAL = 0.05
            cfg.PERC_TEST = 0.05
            cfg.TRAIN_PERC_REAL = 0.0

    # Checks
    if cfg.PERC_TRAIN + cfg.PERC_VAL + cfg.PERC_TEST != 1.0:
        raise Exception("--- ERROR: Dataset splits must equal 1.0")
    if cfg.TRAIN_PERC_REAL < 0.0 or cfg.TRAIN_PERC_REAL > 1.0:
        raise Exception("--- ERROR: Percentage of data that is real for "
                        "training split must be b/w 0.0 & 1.0")

    ############################################
    ############################################
    #       BEGIN
    ############################################
    ############################################



    #######################
    # INPUTS

    # Load inputs via config file ... # TODO - If needed?

    out.output_debug_info(os.path.join(cfg.BASE_PATH, "_REF_run_info.txt"),
                          [cfg._info_model, cfg._info_dataset,
                           cfg._info_results])


    # TODO - how to save off examples of images

    # TODO - Keras.ImageDataGenerator() is debricated. Convert to other methods
    #  such as the 'tf.keras.utils.image_dataset_from_directory and transforming
    #  via tf.data.Dataset preprocessing layers.
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

    # Must use keras version 2.13.1 or earlier

    train_datagen = ImageDataGenerator(
        rotation_range=cfg._2D_aug['rotation_range'],
        width_shift_range=cfg._2D_aug['width_shift_range'],
        height_shift_range=cfg._2D_aug['height_shift_range'],
        brightness_range=cfg._2D_aug['brightness_range'],
        shear_range=cfg._2D_aug['shear_range'],
        zoom_range=cfg._2D_aug['zoom_range'],
        fill_mode=cfg._2D_aug['fill_mode'],
        horizontal_flip=cfg._2D_aug['horizontal_flip'],
        vertical_flip=cfg._2D_aug['vertical_flip'],
        rescale=cfg._2D_aug['rescale'],
        data_format=cfg._2D_aug['data_format'],
    )
    #train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    #######################
    # Load data from dataframe for reference

    def decide_num_images(df_data: pd.DataFrame, desired_num_imgs: int):
        if desired_num_imgs > len(df_data):
            raise Exception(
                "*** ERROR: You have requested too much data for this "
                f"dataset. "
                f"Request: {desired_num_imgs}, Dataset: {len(df_data)}")
        else:
            num_images = desired_num_imgs

        return int(num_images)

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
            print("*** WARN: Number of requested samples is zero. "
                  "Returning empty Dataframe")
            df = pd.DataFrame(columns=df_data.keys())

        return df


    # Load all datasets
    if CASE_FLAG == "weld":
        cfg.CLASS_MAP_ = {'Good': 0,
                          'Defect': 1}
        abs_path_real = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project\\" \
                        "_data\\welding\\real\\df_master_all_real.csv"
        df_master = pd.read_csv(abs_path_real, header=0, index_col=None)

        num_total_images = int(len(df_master) - len(cfg.CLASS_MAP_.keys())) \
            if cfg.TOTAL_NUMBER_OF_IMAGES > len(df_master) \
            else cfg.TOTAL_NUMBER_OF_IMAGES
        cfg.STEPS_PER_EPOCH_TRAIN = int(
            num_total_images * cfg.PERC_TRAIN / cfg.BATCH_SIZE) - 1
        cfg.STEPS_PER_EPOCH_VAL = int(
            num_total_images * cfg.PERC_VAL / cfg.BATCH_SIZE) - 1
        cfg._info_model["NumberOfEpochSteps"] = [cfg.STEPS_PER_EPOCH_TRAIN,
                                                 cfg.STEPS_PER_EPOCH_VAL]

        keys = df_master.keys()
        x = df_master[keys[0]].tolist()
        y = df_master[keys[1]].tolist()
        _, x, _, y = train_test_split(x, y,
                                      test_size=int(num_total_images),
                                      stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                          test_size=(
                                                                      cfg.PERC_TEST + cfg.PERC_VAL),
                                                          stratify=y)
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,
                                                        test_size=cfg.PERC_TEST / (
                                                                    cfg.PERC_TEST + cfg.PERC_VAL),
                                                        stratify=y_val)

        df_train = pd.DataFrame({keys[0]: x_train, keys[1]: y_train})
        df_val = pd.DataFrame({keys[0]: x_val, keys[1]: y_val})
        df_test = pd.DataFrame({keys[0]: x_test, keys[1]: y_test})

        print("*** DEBUG: Dataset Gathered ...\n\t"
              f"Expected:\t{num_total_images * cfg.PERC_TRAIN}/"
              f"{num_total_images * cfg.PERC_VAL}/"
              f"{num_total_images * cfg.PERC_TEST}\n\t"
              f"Actual:\t{len(df_train)}/{len(df_val)}/{len(df_test)}")

        ##
        # Check
        if cfg.STEPS_PER_EPOCH_TRAIN * cfg.BATCH_SIZE > len(df_train):
            num_planned = cfg.STEPS_PER_EPOCH_TRAIN * cfg.BATCH_SIZE
            raise Exception(
                "*** ERROR: Specified training dimensions exceed the "
                "loaded amount of TRAINING data: "
                f"{num_planned} vs {len(df_train)}.")

        if cfg.STEPS_PER_EPOCH_VAL * cfg.BATCH_SIZE > len(df_val):
            num_planned = cfg.STEPS_PER_EPOCH_VAL * cfg.BATCH_SIZE
            raise Exception(
                "*** ERROR: Specified training dimensions exceed the "
                "loaded amount of VALIDATION data: "
                f"{num_planned} vs {len(df_val)}.")

        for label in df_master[keys[1]].unique():
            if label not in list(cfg.CLASS_MAP_.keys()):
                raise Exception(
                    f"*** ERROR: Unknown class {label} found in df_master")

        cfg._info_model["Number of REAL Images"] = len(df_master)
        cfg._info_model["Number of SYNTH Images"] = 0

        out.output_dataset_csv(
            path=os.path.join(cfg.BASE_PATH, "img_list_TRAIN.csv"),
            df=df_train)
        out.output_dataset_csv(
            path=os.path.join(cfg.BASE_PATH, "img_list_VAL.csv"),
            df=df_val)
        out.output_dataset_csv(
            path=os.path.join(cfg.BASE_PATH, "img_list_TEST.csv"),
            df=df_test)

    elif CASE_FLAG == "Cats_Dogs":
        cfg.CLASS_MAP_ = {'dog': 0,
                            'cat': 1}

        abs_path_real = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project\\" \
                        "_data\\cats_dogs\\real_imagenet\\abs_cats_dogs.csv"
        df_real_dataset = pd.read_csv(abs_path_real, header=0, index_col=None)

        abs_path_synth = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project\\" \
                         "_data\\cats_dogs\\synth\\" \
                         "2024_02_28-21_30_51-pipeline\\" \
                         "synth-cats_dogs-corrected.csv"
        df_synth01 = pd.read_csv(abs_path_synth, header=0, index_col=None)
        abs_path_synth = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project\\" \
                         "_data\\cats_dogs\\synth\\" \
                         "2024_03_02-07_46_04-pipeline\\" \
                         "synth-cats_dogs-corrected.csv"
        df_synth02 = pd.read_csv(abs_path_synth, header=0, index_col=None)
        # abs_path_synth = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project\\" \
        #                  "_data\\cats_dogs\\synth\\" \
        #                  "2024_03_06-22_50_49-pipeline\\" \
        #                  "synth-cats_dogs-corrected.csv"
        # df_synth03 = pd.read_csv(abs_path_synth, header=0,
        #                                  index_col=None)
        # abs_path_synth = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project\\" \
        #                  "_data\\cats_dogs\\synth\\" \
        #                  "2024_03_07-06_05_20-pipeline\\" \
        #                  "synth-cats_dogs-corrected.csv"
        # df_synth04 = pd.read_csv(abs_path_synth, header=0,
        #                                  index_col=None)

        total_synth_images = len(df_synth01) + \
                                len(df_synth02) #+ \
                                # len(df_synth03) + \
                                # len(df_synth04)
        cfg._info_model["Number of SYNTH Images available"] = total_synth_images

        sampling_ratio = [ 1.0, 1.0
                          #   len(df_synth01) / total_synth_images,
                          # len(df_synth02) / total_synth_images#,
                          # len(df_synth03) / total_synth_images,
                          # len(df_synth04) / total_synth_images
                          ]

        df_cats = pd.concat([
                    df_synth01[df_synth01["class"] == "cat"].sample(
                                                    frac=sampling_ratio[0]),
                    df_synth02[df_synth02["class"] == "cat"].sample(
                                                    frac=sampling_ratio[1])#,
                    # df_synth03[df_synth03["class"] == "cat"].sample(
                    #                                 frac=sampling_ratio[2]),
                    # df_synth04[df_synth04["class"] == "cat"].sample(
                    #                                 frac=sampling_ratio[3]),
                ])
        df_dogs = pd.concat([
                    df_synth01[df_synth01["class"] == "dog"].sample(
                                                    frac=sampling_ratio[0]),
                    df_synth02[df_synth02["class"] == "dog"].sample(
                                                    frac=sampling_ratio[1])#,
                    # df_synth03[df_synth03["class"] == "dog"].sample(
                    #                                 frac=sampling_ratio[2]),
                    # df_synth04[df_synth04["class"] == "dog"].sample(
                    #                                 frac=sampling_ratio[3]),
                ])
        df_synth_dataset = pd.concat([df_cats, df_dogs])

        # Determine how many pictures are needed for the REAL & SYNTH portions
        perc_train_real = cfg.PERC_TRAIN * cfg.TRAIN_PERC_REAL
        desired_num_real = int(cfg.TOTAL_NUMBER_OF_IMAGES * (perc_train_real +
                                                             cfg.PERC_VAL +
                                                             cfg.PERC_TEST))
        num_real = decide_num_images(df_real_dataset, desired_num_real)
        df_real = binary_sample_dataset(df_real_dataset, num_real)

        desired_num_synth = int(cfg.TOTAL_NUMBER_OF_IMAGES * \
                                cfg.PERC_TRAIN * (1 - cfg.TRAIN_PERC_REAL))
        num_synth = decide_num_images(df_synth_dataset, desired_num_synth)
        df_synth = binary_sample_dataset(df_synth_dataset,
                                         desired_samples=num_synth)

        # CHECK
        if len(df_real) + len(df_synth) > cfg.TOTAL_NUMBER_OF_IMAGES and \
                len(df_real) + len(df_synth) < cfg.TOTAL_NUMBER_OF_IMAGES * 0.98:
            raise Exception("--- ERROR: Double check splitting of data. Unequal "
                            "number of images versus what is desired: "
                            f"{len(df_real) + len(df_synth)} vs "
                            f"{cfg.TOTAL_NUMBER_OF_IMAGES}")

        # Master file of all of the needed data
        df_master = pd.concat([df_real, df_synth])

        def assign_data_splits(df_real: pd.DataFrame, df_synth: pd.DataFrame,
                               split_perc: list, real_synth_ratio: list):
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
                    stratify=df_real[keys[1]].tolist())

                # VAL_REAL & TEST_REAL - Take all remaining samples for VAL & TEST sets
                val_test_split = split_perc[2] / (split_perc[2] + split_perc[1])
                x_val, x_test, y_val, y_test = train_test_split(
                    x_val,
                    y_val,
                    test_size=val_test_split,
                    stratify=y_val)

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
                                    stratify=df_real[keys[1]].tolist())

                # Since PERC_TRAIN_REAL == 0.0, then training data is all synth
                x_train = []
                df_train = df_synth


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

            return df_train, df_val, df_test, [total_real, total_synth]

        # Assign Train/Val/Test splits
        df_train, df_val, df_test, data_ratio = assign_data_splits(
            df_real=df_real,
            df_synth=df_synth,
            split_perc=[cfg.PERC_TRAIN,
                        cfg.PERC_VAL,
                        cfg.PERC_TEST],
            real_synth_ratio=[cfg.TRAIN_PERC_REAL,
                              cfg.VAL_PERC_REAL]
        )
        cfg._info_model["Number of REAL Images"] = data_ratio[0]
        cfg._info_model["Number of SYNTH Images"] = data_ratio[1]

        ##
        # Check
        cfg.STEPS_PER_EPOCH_TRAIN = int(len(df_master) * cfg.PERC_TRAIN /
                                        cfg.BATCH_SIZE)
        cfg.STEPS_PER_EPOCH_VAL = int(len(df_master) * cfg.PERC_VAL /
                                      cfg.BATCH_SIZE)
        cfg._info_model["NumberOfEpochSteps"] = [cfg.STEPS_PER_EPOCH_TRAIN,
                                                 cfg.STEPS_PER_EPOCH_VAL]

        if cfg.STEPS_PER_EPOCH_TRAIN * cfg.BATCH_SIZE > len(df_train):
            num_planned = cfg.STEPS_PER_EPOCH_TRAIN * cfg.BATCH_SIZE
            raise Exception("*** ERROR: Specified training dimensions exceed the "
                            "loaded amount of TRAINING data: "
                            f"{num_planned} vs {len(df_train)}.")

        if cfg.STEPS_PER_EPOCH_VAL * cfg.BATCH_SIZE > len(df_val):
            num_planned = cfg.STEPS_PER_EPOCH_VAL * cfg.BATCH_SIZE
            raise Exception("*** ERROR: Specified training dimensions exceed the "
                            "loaded amount of VALIDATION data: "
                            f"{num_planned} vs {len(df_val)}.")

        for label in df_master[df_master.keys()[1]].unique():
            if label not in list(cfg.CLASS_MAP_.keys()):
                raise Exception(
                    f"*** ERROR: Unknown class {label} found in df_master")

        out.output_dataset_csv(
            path=os.path.join(cfg.BASE_PATH, "img_list_TRAIN.csv"),
            df=df_train)
        out.output_dataset_csv(path=os.path.join(cfg.BASE_PATH, "img_list_VAL.csv"),
                               df=df_val)
        out.output_dataset_csv(
            path=os.path.join(cfg.BASE_PATH, "img_list_TEST.csv"),
            df=df_test)

    #######################
    # Apply 2D augmentations to loaded data

    # Train Generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=None,
        x_col=df_train.columns[0],
        y_col=list(df_train.columns[1:])[0],
        classes=list(cfg.CLASS_MAP_.keys()),
        target_size=cfg.IMAGE_SIZE[:2],
        batch_size=cfg.BATCH_SIZE,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        directory=None,
        x_col=df_val.columns[0],
        y_col=list(df_val.columns[1:])[0],
        classes=list(cfg.CLASS_MAP_.keys()),
        target_size=cfg.IMAGE_SIZE[:2],
        batch_size=cfg.BATCH_SIZE,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test,
        directory=None,
        x_col=df_test.columns[0],
        y_col=list(df_test.columns[1:])[0],
        classes=list(cfg.CLASS_MAP_.keys()),
        target_size=cfg.IMAGE_SIZE[:2],
        batch_size=cfg.BATCH_SIZE,
        class_mode='binary'
    )

    #######################
    # Define parameters for training and train the model

    train_steps = train_generator.n
    val_steps = val_generator.n
    test_steps = test_generator.n

    print(
        "*** DEBUG: to train 100% of data, steps_per_epoch for current data "
        f"must be: {train_generator.n / cfg.BATCH_SIZE}\n"
        f"\t-current: {cfg.STEPS_PER_EPOCH_TRAIN}")

    checkpoint = ModelCheckpoint(os.path.join(cfg.BASE_PATH,
                                              f"basic_impl_{type}.h5"),
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='auto',
                                 save_freq='epoch')
    metric_to_monitor = 'val_accuracy'
    patience = 8
    early = EarlyStopping(monitor=metric_to_monitor,
                          min_delta=0.0005,
                          patience=patience,
                          verbose=1,
                          mode='auto')

    hist = model.fit(
        train_generator,
        steps_per_epoch=cfg.STEPS_PER_EPOCH_TRAIN,
        epochs=cfg.EPOCHS,
        validation_data=val_generator,
        validation_steps=cfg.STEPS_PER_EPOCH_VAL,
        callbacks=[checkpoint, early]
    )

    out.output_learning_plot(os.path.join(cfg.BASE_PATH, "plot_AccLoss.png"),
                             hist)
    out.output_recall_precision_plot(
        os.path.join(cfg.BASE_PATH, "plot_RecPre.png"),
        hist)

    trained_epochs = len(hist.history[metric_to_monitor])
    best_epoch = hist.history[metric_to_monitor].index(
        max(hist.history[metric_to_monitor]))

    cfg._info_results['Total Epochs Trained'] = best_epoch + 1
    cfg._info_results['accuracy'] = hist.history["accuracy"][best_epoch]
    cfg._info_results['loss'] = hist.history["loss"][best_epoch]
    cfg._info_results['recall'] = hist.history["recall"][best_epoch]
    cfg._info_results['precision'] = hist.history["precision"][best_epoch]
    cfg._info_results['val_accuracy'] = hist.history["val_accuracy"][best_epoch]
    cfg._info_results['val_loss'] = hist.history["val_loss"][best_epoch]
    cfg._info_results['val_recall'] = hist.history["val_recall"][best_epoch]
    cfg._info_results['val_precision'] = hist.history["val_precision"][
        best_epoch]
    out.output_debug_info(os.path.join(cfg.BASE_PATH, "_REF_run_info.txt"),
                          [cfg._info_model, cfg._info_dataset,
                           cfg._info_results])

    #######################
    # Test the fully trained model

    print("\n***** Model Testing *****")
    # TODO - Review this .evaluate() method... Are the five {l, acc, ba, re, pre} ?
    # Test the model against the test data
    scores = model.evaluate(test_generator)
    try:
        cfg._info_results['test_accuracy'] = scores[1]
        cfg._info_results['test_loss'] = scores[0]
        cfg._info_results['test_recall'] = scores[3]
        cfg._info_results['test_precision'] = scores[4]
        if cfg.NUM_CLASSES == 2:
            cfg._info_results['test_binary_accuracy'] = scores[2]
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
                                            'pred_value'])

    # Perform the manual test by manually calling .predict()
    for i in range(len(df_test)):
        if i % int(len(df_test) / 10) == 0:
            print(f"\t\t\t{int(i / len(df_test) * 100)}% complete...")
        img_path = df_test.iloc()[i, 0]
        label = df_test.iloc()[i, 1]

        img = keras.utils.load_img(img_path, target_size=cfg.IMAGE_SIZE)
        img = process_test_image(img, False)
        prediction, pred = make_binary_prediction(model, cfg.BINARY_THRESHOLD,
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
                                                           pred[0][0]]
    out.output_dataset_csv(path=os.path.join(cfg.BASE_PATH,
                                             "img_list_TEST_RESULTS.csv"),
                           df=df_test_results)

    # Analyze the manual test results
    num_correct = len(df_test_results[df_test_results['result'] == 1])
    cfg._info_results['test_Manual_accuracy'] = \
        round(num_correct / len(df_test_results), 2)
    cfg._info_results['test_Manual_TP_TN'] = num_correct
    out.output_debug_info(os.path.join(cfg.BASE_PATH, "_REF_run_info.txt"),
                          [cfg._info_model, cfg._info_dataset,
                           cfg._info_results])

    print(f"--- Manual Test Score: "
          f"Accuracy={round(num_correct / len(df_test_results), 3)} "
          f"= ({num_correct})")

    print("***** Testing Complete")

    out.output_Confusion_Mat_Heatmap(os.path.join(cfg.BASE_PATH,
                                                  "Test_Results_Heatmap.png"),
                                     df_true=df_test_results['actual'],
                                     df_preds=df_test_results['pred'],
                                     classes=list(df_test["class"].unique()))

    # (For multi-script execution) Reset keras backend session to release variables
    # keras.backend.clear_session()
    del model, train_generator, checkpoint

    return cfg._info_results['test_Manual_accuracy'],   \
            cfg._info_results['test_binary_accuracy'],     \
            cfg._info_results['accuracy']