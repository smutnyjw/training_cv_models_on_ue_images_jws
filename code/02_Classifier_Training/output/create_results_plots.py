'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import math

def output_acc_plot(df: pd.DataFrame, path: str):
    print("*** Outputting a Accuracy vs Synth image plot")

    ###
    # output a line plot detailing the change in Accuracy compared to
    # number of synthetic training images

    acc = list(df['accuracy'])
    test_acc = list(df['test_Manual_accuracy'])
    num_real = list(df['Number of REAL Training Images'])
    num_synth = list(df['Number of SYNTH Images'])

    if df['id'].min() == 0:
        x = num_synth
        xLabel = "Number of Synthetic Images"
        constant_samples = f"{num_real[0]} Real"

    else:
        x = num_real
        xLabel = "Number of Real Training Images"
        constant_samples = f"{num_synth[0]} Synthetic"

    ##

    if min(acc) < 0.5 or min(test_acc) < 0.5:
        y_min = 0.0
        y_max = 1.0
    elif 0.5 < min(acc) < 0.8 or 0.5 < min(test_acc) < 0.8:
        y_min = 0.5
        y_max = 1.0
    elif 0.8 < min(acc) < 0.9 or 0.8 < min(test_acc) < 0.9:
        y_min = 0.8
        y_max = 1.0
    elif 0.9 < min(acc) or 0.9 < min(test_acc):
        y_min = 0.9
        y_max = 1.0

    ##

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(xLabel)
    ax1.set_ylabel("Accuracy")
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(y_min, y_max)
    ax1.plot(x, acc, 'b-', label='Acc')
    ax1.plot(x, test_acc, 'b--', label='Test_Acc')

    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='lower right')

    plt.title(f"Model Training Accuracy - {constant_samples} Images")
    plt.savefig(path)  # "output/basic_impl_model_training_results.png")
    plt.close()
    plt.cla()


def output_acc_loss_plot(df: pd.DataFrame, path: str):
    print("*** Outputting a Accuracy + Loss vs Synth image plot")

def output_loss_plot(df: pd.DataFrame, path: str):
    print("*** Outputting a Loss vs Synth image plot")

    ###
    # output a line plot detailing the change in Loss compared to
    # number of synthetic training images

    loss = list(df['loss'])
    test_loss = list(df['test_loss'])
    num_real = list(df['Number of REAL Training Images'])
    num_synth = list(df['Number of SYNTH Images'])

    if df['id'].min() == 0:
        x = num_synth
        xLabel = "Number of Synthetic Images"
        constant_samples = f"{num_real[0]} Real"

    else:
        x = num_real
        xLabel = "Number of Real Training Images"
        constant_samples = f"{num_synth[0]} Synthetic"

    ###

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(xLabel)
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_ylim(0.0, 2.0)
    ax1.plot(x, loss, 'r-', label='loss')
    ax1.plot(x, test_loss, 'r--', label='Test_Loss')

    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='upper right')

    plt.title(f"Model Training Loss - {constant_samples} Images")
    plt.savefig(path)  # "output/basic_impl_model_training_results.png")
    plt.close()
    plt.cla()

def run_logic(CASE: str, MODEL: str, A_ID: int):
    csv_input = f"{MODEL}\\{CASE}_test_results.csv"
    path = "C:\\Users\\johns\\PycharmProjects\\VT\\" \
           "training_cv_models_on_ue_images_jws\\code\\" \
           "02_Classifier_Training\\output"
    inc_synth = range(0,16)
    inc_real = range(100,116)
    TRAIN_INC_RANGE = inc_real if A_ID else inc_synth

    df_csv = pd.read_csv(os.path.join(path, csv_input))

    for i, model in enumerate(list(df_csv['Model'].unique())):
        for j, case in enumerate(list(df_csv['Classes'].unique())):
            type = "cat_dog" if "cat" in case else "weld"
            inc = "IncSynth" if TRAIN_INC_RANGE[0] == 0 else "IncReal"



            #for k in list(df_csv['id'].unique()):
            df_instance = df_csv.query(f'Model == \"{model}\" and '
                                        f'Classes == \"{case}\"')

            if df_instance.empty is False:
                # Average the results for each 'id'
                results_dict = {'id': [],
                                 'accuracy': [],
                                 'test_Manual_accuracy': [],
                                 'loss': [],
                                 'test_loss': [],
                                 'Number of REAL Training Images': [],
                                 'Number of SYNTH Images': [],
                                 'Dataset Breakdown': []
                                }

                for id in TRAIN_INC_RANGE:
                    df_r = df_instance.query(f'id == {id}')

                    if len(list(df_r['accuracy'])) == 0:
                        print(f"---- Skip {model} {case} inc_{id}")
                        break

                    results_dict['id'].append(int(id))
                    results_dict['accuracy'].append(df_r['accuracy'].mean())
                    results_dict['test_Manual_accuracy'].append(
                                    df_r['test_Manual_accuracy'].mean())
                    results_dict['loss'].append(df_r['loss'].mean())
                    results_dict['test_loss'].append(df_r['test_loss'].mean())
                    results_dict['Number of REAL Training Images'].append(
                            int(df_r['Number of REAL Images'].mean()))
                    results_dict['Number of SYNTH Images'].append(
                            int(df_r['Number of SYNTH Images'].mean()))
                    results_dict['Dataset Breakdown'].append(
                        list(df_r['Dataset Breakdown (TRAIN/VAL/TEST)']))

                df_avg = pd.DataFrame.from_dict(results_dict)
                df_avg = df_avg.sort_values(by=["id"])
                # Output Plots

                output_acc_plot(df_avg, os.path.join(path,
                                f"{model}_{type}_{inc}_acc_plot.png"))

                output_loss_plot(df_avg, os.path.join(path,
                                f"{model}_{type}_{inc}_loss_plot.png"))


if __name__=="__main__":
    CASE = ["cats_dogs", "weld"]
    MODEL = ["vgg16", "mobilenet"]
    INCREMENTS = [0, 1]

    CASE = ["weld"]


    for model in MODEL:
        for case in CASE:
            for increment_type in INCREMENTS:
                run_logic(CASE=case, MODEL=model, A_ID=increment_type)