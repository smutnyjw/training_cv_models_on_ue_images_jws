'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import os, sys
import json
import pandas as pd
from datetime import datetime

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input


def build_vgg16_model(img_size):

    num_classes = 1000

    # Load base VGG16
    vgg16_model = keras.applications.VGG16(
        input_shape=img_size,
        include_top=False,
        weights="imagenet"
    )

    ###
    # OUTPUT model artifacts
    # path_vgg16 = os.path.join(base_path,
    #                             f"vgg16_{num_classes}_trainable_layers.txt")
    # with open(path_vgg16, 'w') as sys.stdout:
    #     vgg16_model.summary(show_trainable=True)
    #
    # sys.stdout = sys.__stdout__

    return vgg16_model


def build_mobilenet_model(img_size):

    num_classes = 1000

    # Load base MobileNetV3
    # https://keras.io/api/applications/mobilenet/
    mobilenet_model = keras.applications.MobileNetV3Small(
        input_shape=img_size,       #default is (224,224,3)
        alpha=1.0,    # default=1    Causes error if not 1
        minimalistic=False,
        include_top=True,
        weights="imagenet"
    )

    ####
    # OUTPUT model artifacts
    # path_mobilenet = os.path.join(base_path,
    #                               f"mobilenet_{num_classes}_trainable_layers.txt")
    # with open(path_mobilenet, 'w') as sys.stdout:
    #     mobilenet_model.summary(show_trainable=True)
    #
    # sys.stdout = sys.__stdout__

    return mobilenet_model

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


def output_dataset_csv(path, df: pd.DataFrame):
    # Output csv log
    try:
        df.to_csv(path)
    except NameError:
        print(f"**** WARN: Dataframes to {path} was not defined")


def run_model_for_df(df: pd.DataFrame, model, out_path: str, data_type: str,
                     img_size, CLASS_MAP_: dict):
    df_test_results = pd.DataFrame(columns=['filepath',
                                            'pred', 'actual', 'result',
                                            'confidence'])

    for i in range(len(df)):
        if i % int(len(df) / 10) == 0:
            print(f"\t\t\t{int(i / len(df) * 100)}% complete...")
        img_path = df.iloc()[i, 0]
        label = df.iloc()[i, 1]

        img = keras.utils.load_img(img_path, target_size=img_size)
        img = process_test_image(img, False)
        prediction, confidence = make_binary_prediction(model,
                                                        0.5,
                                                        img)

        # Decode the prediction
        result = -1
        for key in CLASS_MAP_.keys():
            if CLASS_MAP_[key] == prediction:
                prediction = key
                result = 1 if prediction == label else 0
                break

        # store results in a dataframe
        df_test_results.loc[len(df_test_results.index)] = [img_path,
                                                           prediction,
                                                           label,
                                                           result,
                                                           confidence[0][0]]
    output_dataset_csv(path=os.path.join(out_path,
                                             f"TEST_RESULTS_{data_type}.csv"),
                           df=df_test_results)


def setup_output_dir(base_path, type, case: str):
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
    path = os.path.join(base_path, type,
                        f"{start}-{type}-{case}-conf")

    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

    return path


if __name__=="__main__":

    # Settings and Paths
    cat_dog_real_path = ""
    cat_dog_synth_path = ""
    base_path = ""
    img_size = (224, 224, 3)

    # Initialize each model for 'test' only. No training
    mn_model = build_mobilenet_model(img_size)
    vgg_model = build_vgg16_model(img_size)

    # load data to be tested
    df_test_real = pd.read_csv(cat_dog_real_path)
    df_test_synth = pd.read_csv(cat_dog_synth_path)

    ##################
    # Perform the manual test by manually calling .predict().
    CLASS_MAP = {'dog': 0, 'cat': 1}

    results_path = os.path.join(base_path, "vgg16")
    setup_output_dir(base_path=results_path, type="VGG16", case="Cat_Dog")

    run_model_for_df(df=df_test_real, model=vgg_model, out_path=results_path,
                     data_type="real", img_size=img_size, CLASS_MAP_=CLASS_MAP)
    run_model_for_df(df=df_test_synth, model=vgg_model, out_path=results_path,
                     data_type="synth", img_size=img_size, CLASS_MAP_=CLASS_MAP)

    results_path = os.path.join(base_path, "mobilenet")
    setup_output_dir(base_path=results_path, type="MobileNet", case="Cat_Dog")

    run_model_for_df(df=df_test_real, model=mn_model, out_path=results_path,
                     data_type="real", img_size=img_size, CLASS_MAP_=CLASS_MAP)
    run_model_for_df(df=df_test_synth, model=mn_model, out_path=results_path,
                     data_type="synth", img_size=img_size, CLASS_MAP_=CLASS_MAP)





