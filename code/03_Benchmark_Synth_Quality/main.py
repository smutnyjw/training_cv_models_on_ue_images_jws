'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.applications import vgg16, mobilenet_v3


def build_vgg16_model(img_size):

    num_classes = 1000

    # Load base VGG16
    vgg16_model = keras.applications.VGG16(
        # input_shape=img_size,
        # include_top=False,
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
        # input_shape=img_size,       #default is (224,224,3)
        # alpha=1.0,    # default=1    Causes error if not 1
        # minimalistic=False,
        # include_top=False,
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


def process_test_image(img, show_image, opt: str):
    if show_image:
        print(img)
        plt.imshow(img)

    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    if opt.lower == "vgg16":
        x = vgg16.preprocess_input(x)
    else:
        x = mobilenet_v3.preprocess_input(x)

    return x


def output_dataset_csv(df: pd.DataFrame, path: str):
    # Output csv log
    try:
        df.to_csv(path)
    except NameError:
        print(f"**** WARN: Dataframes to {path} was not defined")


def output_distribution_plot(df, path: str, data_type: str, model_type: str):

    bins = range(0, 100, 5)
    df_top3 = df.query("top_gram >= 0")
    dist1 = df_top3['confidence'] * 100
    df_outside = df.query("top_gram == -1")
    dist2 = df_outside['confidence'] * 100

    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=False, tight_layout=True)
    ax.hist(dist1, bins=bins)
    #ax[1].hist(dist2, bins=bins)

    fig.suptitle(f"Confidence Score Distribution "
            f"for {data_type} Cats & Dogs", fontsize=14, fontweight="bold")
    perc_in_top3 = "{0:.1f}".format(len(df_top3) / len(df) * 100)

    subtitle = f"Audited by a pre-trained {model_type} model with" \
                f" Imagenet weights,\n3-Shot Classification Rate = " \
                f"{perc_in_top3}% ({len(df_top3)}/{len(df)})"
    ax.set_title(subtitle, fontsize=12, color="gray")
    #ax[1].set_title(f"Actual class not in Top3 ({len(df_outside)}/{len(df)})")
    ax.set_xlabel("Confidence Score (%)", fontsize=14)
    ax.set_ylabel("Number of Images", fontsize=14)
    ax.set_ylim([0, 2200])

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.savefig(path)
    plt.close()
    plt.cla()


def run_model_for_df(df: pd.DataFrame, model, out_path: str, data_type: str,
                     img_size, CLASS_MAP_: dict, model_type: str,
                     load_csv_path):
    print(f"**** Run for {data_type} images - {model_type}")

    type = out_path.split('\\')[-1]
    df_test_results = pd.DataFrame(columns=['filepath',
                                            'actual', 'top_gram', 'confidence',
                                            'Top1', 'Top2', 'Top3'])
    type_str = f"{model_type}-{data_type}"
    csv_filename = f"Confidence_RESULTS-{type_str}.csv"
    output_csv_path = os.path.join(out_path, csv_filename)

    #######
    #
    #######

    if load_csv_path is False:
        for i in range(len(df)):
            if i % int(len(df) / 10) == 0:
                print(f"\t\t\t{int(i / len(df) * 100)}% complete...")
            img_path = df.iloc()[i, 0]
            label = df.iloc()[i, 1]

            # Get prediction confidence scores for each image
            img = keras.utils.load_img(img_path, target_size=img_size)
            img = process_test_image(img, False, type)
            conf = model.predict(img, verbose=0)

            # Get top 3 confidence scores
            _3_gram= []
            if type.lower() == "vgg16":
                decoded = vgg16.decode_predictions(conf, top=3)[0]
            else:
                decoded = mobilenet_v3.decode_predictions(conf, top=3)[0]

            for item in decoded:
                _3_gram.append(item[1])
            #print(f"{type.lower()} Predictions: {_3_gram}")


            # Determine if the actual label is in the top 3
            valid_keys = CLASS_MAP_[label]
            # find_conf = np.copy(conf[0])
            # result = -1
            # for id in range(3):
            #     if np.argmax(find_conf) in valid_keys:
            #         result = id
            #         break
            #     else:
            #         find_conf[np.argmax(find_conf)] = -1
            cs = np.copy(conf[0])
            cs.sort()
            top_three_indexs = [np.argwhere(conf[0] == cs[-1])[0][0],
                                np.argwhere(conf[0] == cs[-2])[0][0],
                                np.argwhere(conf[0] == cs[-3])[0][0]]

            result = -1
            confidence = 0.0
            for i, top_id in enumerate(top_three_indexs):
                if top_id in valid_keys:
                    result = i
                    confidence = conf[0][top_id]
                    break

            # Get the top confidence score out of valid keys if not in top 3
            if result == -1:
                for key in valid_keys:
                    if conf[0][key] > confidence:
                        confidence = conf[0][key]

            # Hard cap the confidence score
            confidence = 1 if confidence > 1 else confidence

            # store results in a dataframe
            df_test_results.loc[len(df_test_results.index)] = [img_path,
                                                               label,
                                                               result,
                                                               confidence,
                                                               _3_gram[0],
                                                               _3_gram[1],
                                                               _3_gram[2]
                                                               ]
            output_dataset_csv(path=output_csv_path, df=df_test_results)
        # End of img for loop
    else:
        # Load the results csv file
        df_test_results = pd.read_csv(load_csv_path)
        print(f"Loaded existing csv\n\t{output_csv_path}")

    output_distribution_plot(df=df_test_results,
                                path=os.path.join(out_path,
                                                f"Dist_Plot-{type_str}.png"),
                                data_type=data_type, model_type=model_type)


def setup_output_dir(base_path, case: str):
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
    type = base_path.split('\\')[-1]
    path = os.path.join(base_path,
                        f"{start}-{type}-{case}-conf")

    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

    return path


if __name__=="__main__":

    # Settings and Paths
    cat_dog_real_path = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project" \
                        "\\_data\\cats_dogs\\real_imagenet\\abs_cats_dogs.csv"
    cat_dog_synth_path = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project" \
                        "\\_data\\cats_dogs\\synth\\full_synth_cat_dog.csv"
    base_path = "C:\\Users\\johns\\PycharmProjects\\VT" \
                "\\training_cv_models_on_ue_images_jws\\code" \
                "\\03_Benchmark_Synth_Quality\\output"
    img_size = (224, 224, 3)

    # Initialize each model for 'test' only. No training
    mn_model = build_mobilenet_model(img_size)
    vgg_model = build_vgg16_model(img_size)

    # load data to be tested
    df_test_real = pd.read_csv(cat_dog_real_path)
    df_test_synth = pd.read_csv(cat_dog_synth_path)

    # df_test_real = df_test_real.sample(100)
    # df_test_synth = df_test_synth.sample(100)

    ##################
    # Perform the manual test by manually calling .predict().
    #   https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
    #   Dog iffy: 275
    #   Cat iffy: 286, 287, 383, 387
    CLASS_MAP = {'dog': list(range(151, 268)),
                 'cat': [281, 282, 283, 284, 285, 286, 287, 383, 387]}

    results_path = setup_output_dir(base_path=base_path, case="Cat_Dog")

    ##
    load_csv_path1 = "C:\\Users\\johns\\PycharmProjects\\VT" \
                     "\\training_cv_models_on_ue_images_jws\\code" \
                     "\\03_Benchmark_Synth_Quality\\output\\" \
                     "Confidence_RESULTS-vgg16-real.csv"
    load_csv_path2 = "C:\\Users\\johns\\PycharmProjects\\VT" \
                     "\\training_cv_models_on_ue_images_jws\\code" \
                     "\\03_Benchmark_Synth_Quality\\output" \
                     "\\Confidence_RESULTS-vgg16-synth.csv"
    load_csv_path3 = "C:\\Users\\johns\\PycharmProjects\\VT" \
                     "\\training_cv_models_on_ue_images_jws\code" \
                     "\\03_Benchmark_Synth_Quality\\output" \
                     "\\Confidence_RESULTS-mobilenet-real.csv"
    load_csv_path4 = "C:\\Users\\johns\\PycharmProjects\\VT" \
                     "\\training_cv_models_on_ue_images_jws\\code" \
                     "\\03_Benchmark_Synth_Quality\\output" \
                     "\\Confidence_RESULTS-mobilenet-synth.csv"

    # VGG16
    run_model_for_df(df=df_test_real, model=vgg_model, out_path=results_path,
                     data_type="Real", img_size=img_size, CLASS_MAP_=CLASS_MAP,
                     model_type="vgg16", load_csv_path=load_csv_path1)
    run_model_for_df(df=df_test_synth, model=vgg_model, out_path=results_path,
                     data_type="Synthetic", img_size=img_size,
                     CLASS_MAP_=CLASS_MAP,
                     model_type="vgg16", load_csv_path=load_csv_path2)
    # MobileNet
    run_model_for_df(df=df_test_real, model=mn_model, out_path=results_path,
                     data_type="Real", img_size=img_size, CLASS_MAP_=CLASS_MAP,
                     model_type="mobilenet", load_csv_path=load_csv_path3)
    run_model_for_df(df=df_test_synth, model=mn_model, out_path=results_path,
                     data_type="Synthetic", img_size=img_size,
                     CLASS_MAP_=CLASS_MAP,
                     model_type="mobilenet", load_csv_path=load_csv_path4)





