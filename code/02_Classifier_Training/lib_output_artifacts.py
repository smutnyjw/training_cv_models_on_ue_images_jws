'''
File:   lib_output_artifacts.py
Author: John Smutny
Date:   03/26/2024
Description: 
    Library file used by 'run_model.py' that will output various artifacts
    involved with the training of a Machine Learning model.
Other Notes:

'''
import math
import sys
import os

import keras
import matplotlib.pyplot as plt
import pandas

import seaborn as sns


def output_plots(base_path, history, rs_ratio, num_init_epochs):
    output_learning_plot(os.path.join(base_path, "plot_AccLoss.png"),
                            history,
                            rs_ratio,
                            num_init_epochs,
                            True)
    output_learning_acc_only_plot(
        os.path.join(base_path, "plot_Acc.png"),
        history,
        rs_ratio,
        num_init_epochs,
        True)
    output_learning_loss_only_plot(
        os.path.join(base_path, "plot_Loss.png"),
        history,
        rs_ratio,
        num_init_epochs)
    output_recall_precision_plot(
        os.path.join(base_path, "plot_RecPre.png"),
        history,
        rs_ratio,
        num_init_epochs)

    output_learning_plot(
        os.path.join(base_path, "plot_AccLoss-noValAcc.png"),
        history,
        rs_ratio,
        num_init_epochs, False)
    output_learning_acc_only_plot(
        os.path.join(base_path, "plot_Acc-NoValAcc.png"),
        history,
        rs_ratio,
        num_init_epochs, False)

def output_model_arch(path, model: keras.Model):
    with open(path, 'w') as sys.stdout:
        model.summary(show_trainable=True)

    sys.stdout = sys.__stdout__


def output_learning_plot(path, hist, rs_ratio, num_init_epochs, output_val_acc):
    # Artifact: PLOT - Training Accuracy & Loss
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(0.0, 1.0)
    ax1.plot(hist.history["accuracy"], 'b-', label='Acc')

    if output_val_acc:
        ax1.plot(hist.history['val_accuracy'], 'b--', label='Val_Acc')

    handles1, labels1 = ax1.get_legend_handles_labels()

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss")
    ax2.tick_params(axis='y', labelcolor='r')
    max_loss = math.ceil(max(hist.history['loss'] + hist.history['val_loss']))
    ax2.set_ylim(0.0, max_loss)

    if num_init_epochs > 0:
        plt.vlines(x=num_init_epochs, colors='g', ymin=0, ymax=max_loss,
                   linestyles='dotted',
                   label="Initial Training")

    ax2.plot(hist.history['loss'], 'r-', label='Loss')
    ax2.plot(hist.history['val_loss'], 'r--', label='Val_Loss')
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(handles1+handles2, labels1+labels2, loc='lower left')

    plt.title(f"Model Training Accuracy & Loss - {rs_ratio}")
    plt.savefig(path) #"output/basic_impl_model_training_results.png")
    plt.close()
    plt.cla()


def output_learning_acc_only_plot(path, hist, rs_ratio,
                                  num_init_epochs, output_val_acc):
    plt.plot(hist.history["accuracy"], 'b-')
    if output_val_acc:
        plt.plot(hist.history['val_accuracy'], 'b--')
        plt.legend(["Acc", "Val_Acc"])
    else:
        plt.legend(["Acc"])

    if num_init_epochs > 0:
        plt.vlines(x=num_init_epochs, colors='g', ymin=0, ymax=1.0,
                   linestyles='dotted',
                   label="Initial Training")

    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title(f"Model Training Accuracy - {rs_ratio}")
    plt.savefig(path)
    plt.close()
    plt.cla()


def output_learning_loss_only_plot(path, hist, rs_ratio, num_init_epochs):
    plt.plot(hist.history["loss"], 'r-')
    plt.plot(hist.history['val_loss'], 'r--')
    max_loss = math.ceil(max(hist.history['loss']+hist.history['val_loss']))
    plt.ylim(0.0, max_loss)

    if num_init_epochs > 0:
        plt.vlines(x=num_init_epochs, colors='g', ymin=0, ymax=max_loss,
                   linestyles='dotted',
                   label="Initial Training")

    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title(f"Model Training Loss - {rs_ratio}")
    plt.legend(["Acc", "Val_Loss"])
    plt.savefig(path)
    plt.close()
    plt.cla()


def output_recall_precision_plot(path, hist, rs_ratio, num_init_epochs):
    # Artifact: PLOT - Training Accuracy & Loss
    plt.plot(hist.history["recall"], 'g-')
    plt.plot(hist.history['val_recall'], 'g--')
    plt.plot(hist.history['precision'], 'm-')
    plt.plot(hist.history['val_precision'], 'm--')

    if num_init_epochs > 0:
        plt.vlines(x=num_init_epochs, ymin=0, ymax=1.0,
                   colors='r', linestyles='dotted',
                   label="Initial Training")

    plt.title(f"Model Training Recall & Precision - {rs_ratio}")
    plt.ylabel("Recall/Precision")
    plt.xlabel("Epoch")
    plt.legend(["Recall", "Val Recall", "Prec", "Val Prec"])
    plt.savefig(path)  # "output/basic_impl_model_training_results.png")
    plt.close()
    plt.cla()

def output_Confusion_Mat_Heatmap(conf_matx, path, rs_ratio, cost, classes):
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
    plt.title(f"Confusion Matrix - Cost per Sample: {cost}\n"
              f"100% Real Test Images - {rs_ratio} R:Syn Training Images")
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
