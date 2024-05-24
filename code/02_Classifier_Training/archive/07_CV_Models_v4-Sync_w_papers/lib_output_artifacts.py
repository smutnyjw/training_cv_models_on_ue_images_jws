'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''
import math
import sys

import keras
import matplotlib.pyplot as plt
import pandas

from sklearn.metrics import confusion_matrix
import seaborn as sns


def output_model_arch(path, model: keras.Model):
    with open(path, 'w') as sys.stdout:
        model.summary(show_trainable=True)

    sys.stdout = sys.__stdout__


def output_learning_plot(path, hist):
    # Artifact: PLOT - Training Accuracy & Loss
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(0.0, 1.0)
    ax1.plot(hist.history["accuracy"], 'b-', label='Acc')
    ax1.plot(hist.history['val_accuracy'], 'b--', label='Val_Acc')
    handles1, labels1 = ax1.get_legend_handles_labels()

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss")
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0.0,
                 math.ceil(max(hist.history['loss']+hist.history['val_loss'])))
    ax2.plot(hist.history['loss'], 'r-', label='Loss')
    ax2.plot(hist.history['val_loss'], 'r--', label='Val_Loss')
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(handles1+handles2, labels1+labels2, loc='lower left')

    plt.title("Model Training Accuracy & Loss")
    plt.savefig(path) #"output/basic_impl_model_training_results.png")
    plt.close()
    plt.cla()


def output_learning_acc_only_plot(path, hist):
    plt.plot(hist.history["accuracy"], 'b-')
    plt.plot(hist.history['val_accuracy'], 'b--')
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title("Model Training Accuracy")
    plt.legend(["Acc", "Val Acc"])
    plt.savefig(path)
    plt.close()
    plt.cla()


def output_recall_precision_plot(path, hist):
    # Artifact: PLOT - Training Accuracy & Loss
    plt.plot(hist.history["recall"], 'g-')
    plt.plot(hist.history['val_recall'], 'g--')
    plt.plot(hist.history['precision'], 'm-')
    plt.plot(hist.history['val_precision'], 'm--')
    plt.title("Model Training Recall & Precision")
    plt.ylabel("Recall/Precision")
    plt.xlabel("Epoch")
    plt.legend(["Recall", "Val Recall", "Prec", "Val Prec"])
    plt.savefig(path)  # "output/basic_impl_model_training_results.png")
    plt.close()
    plt.cla()

def output_Confusion_Mat_Heatmap(path, df_true, df_preds, classes):
    conf_matx = confusion_matrix(df_true, df_preds)
    hm = sns.heatmap(conf_matx,
                     annot=True,
                     annot_kws={"size": 10},
                     fmt='.0f',
                     square=True,
                     cbar=True,
                     xticklabels=classes,
                     yticklabels=classes,
                     linewidth=0.5
                     )
    hm.set(xlabel="Model Prediction", ylabel="Actual")
    plt.savefig(path)
    plt.close()
    plt.cla()


def output_dataset_csv(path, df: pandas.DataFrame):
    # Output csv log
    try:
        df.to_csv(path)
    except NameError:
        print(f"**** WARN: Dataframes to {path} was not defined")


def output_debug_info(path, logs: list):
    '''
    Print out a .to_string() file describing the executed model classifier run.
    :param path:    Location of the new file
    :param logs:    [main, model, dataset, results] collection of dictionaries
    :return:
    '''

    file = open(path, "w")

    for entry in logs:
        file.write("************************************************\n"
                   f"\t\t{entry['HEADER']}\n"
                   "************************************************\n")
        for i in range(1, len(entry.values())):
            file.write(f"\n{list(entry.keys())[i]}: "
                       f"\t{list(entry.values())[i]}")

        file.write("\n\n")

    file.close()
