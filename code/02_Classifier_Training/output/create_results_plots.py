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

def output_acc_plot(df: pd.DataFrame, path: str):
    print("*** Outputting a Accuracy vs Synth image plot")

    ###
    # output a line plot detailing the change in Accuracy compared to
    # number of synthetic training images

    acc = list(df['accuracy'])
    test_acc = list(df['test_Manual_accuracy'])
    num_real = list(df['Number of REAL Images'])
    num_synth = list(df['Number of SYNTH Images'])

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
    ax1.set_xlabel("Number of Synthetic Images")
    ax1.set_ylabel("Accuracy")
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(y_min, y_max)
    ax1.plot(num_synth, acc, 'b-', label='Acc')
    ax1.plot(num_synth, test_acc, 'b--', label='Test_Acc')

    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='lower right')

    plt.title(f"Model Training Accuracy - {num_real[0]} Real Images")
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
    num_real = list(df['Number of REAL Images'])
    num_synth = list(df['Number of SYNTH Images'])

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Number of Synthetic Images")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_ylim(0.0, 2.0)
    ax1.plot(num_synth, loss, 'r-', label='loss')
    ax1.plot(num_synth, test_loss, 'r--', label='Test_Loss')

    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='upper right')

    plt.title(f"Model Training Loss - {num_real[0]} Real Images")
    plt.savefig(path)  # "output/basic_impl_model_training_results.png")
    plt.close()
    plt.cla()

if __name__=="__main__":
    csv_input = "vgg16\\cat_dog_test_results.csv"
    path = "C:\\Users\\johns\\PycharmProjects\\VT\\" \
           "training_cv_models_on_ue_images_jws\\code\\" \
           "02_Classifier_Training\\output"

    df_csv = pd.read_csv(os.path.join(path, csv_input))

    for i, model in enumerate(list(df_csv['Model'].unique())):
        for j, case in enumerate(list(df_csv['Classes'].unique())):
            type = "cat_dog" if "cat" in case else "weld"


            #for k in list(df_csv['id'].unique()):
            df_instance = df_csv.query(f'Model == \"{model}\" and '
                                        f'Classes == \"{case}\"')

            if df_instance.empty is False:
                output_acc_plot(df_instance, os.path.join(path,
                                f"{model}_{type}_acc_plot.png"))

                output_loss_plot(df_instance, os.path.join(path,
                                f"{model}_{type}_loss_plot.png"))


