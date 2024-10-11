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

def output_shadow_acc_plot(df: pd.DataFrame, title_str: str, path: str,
                           y_range: tuple):
    print("*** Outputting a Accuracy shadow vs Synth image plot")

    ###
    # output a line plot detailing the change in Accuracy compared to
    # number of synthetic training images

    acc_mean = list(df['acc_mean'])
    acc_min = list(df['acc_min'])
    acc_max = list(df['acc_max'])

    test_acc_mean = list(df['test_Manual_acc_mean'])
    test_acc_min = list(df['test_Manual_acc_min'])
    test_acc_max = list(df['test_Manual_acc_max'])

    num_real = list(df['Number of REAL Training Images'])
    num_synth = list(df['Number of SYNTH Images'])

    if df['id'].min() == 0:
        x = num_synth
        xLabel = "Number of Synthetic Images"
        subtitle = f"Training Set: {num_real[0]} Real + various Synthetic"

    else:
        x = num_real
        xLabel = "Number of Real Training Images"
        subtitle = f"Training Set: various Real + {num_synth[0]} Synthetic"

    str_trails = f"{df['num'].min()} trials"

    ###

    y_min = y_range[0] - 0.02
    y_max = y_range[1]

    ###

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(xLabel, fontsize=14)
    ax1.set_ylabel("Accuracy", fontsize=14)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(y_min, y_max)

    ax1.plot(x, acc_mean, 'b--', label='Training_Acc')
    ax1.plot(x, test_acc_mean, 'b-', label='Test_Acc')

    ax1.plot(x, test_acc_min, 'b-', alpha=0.01)
    ax1.plot(x, test_acc_max, 'b-', alpha=0.01)
    ax1.fill_between(x, test_acc_min, test_acc_max,
                   interpolate=True, color="blue", alpha=0.1)

    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='lower right', prop={'size': 14})

    fig.suptitle(f"{title_str} - Accuracy - {str_trails}",
                 fontsize=14, fontweight="bold")
    ax1.set_title(subtitle, fontsize=13, color="gray")

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.savefig(path)  # "output/basic_impl_model_training_results.png")
    plt.close()
    plt.cla()


def output_shadow_loss_plot(df: pd.DataFrame, title_str: str, path: str):
    print("*** Outputting a Loss Shadow vs Synth image plot")

    ###
    # output a line plot detailing the change in Accuracy compared to
    # number of synthetic training images

    loss_mean = list(df['loss_mean'])
    loss_min = list(df['loss_min'])
    loss_max = list(df['loss_max'])

    test_loss_mean = list(df['test_loss_mean'])
    test_loss_min = list(df['test_loss_min'])
    test_loss_max = list(df['test_loss_max'])

    num_real = list(df['Number of REAL Training Images'])
    num_synth = list(df['Number of SYNTH Images'])

    if df['id'].min() == 0:
        x = num_synth
        xLabel = "Number of Synthetic Images"
        subtitle = f"Training Set: {num_real[0]} Real + <variable> synth"

    else:
        x = num_real
        xLabel = "Number of Real Training Images"
        subtitle = f"Training Set: <variable> Real + {num_synth[0]} synth"

    str_trails = f"{df['num'].min()} trials"

    ###

    if max(loss_max) > 2.0 or max(test_loss_max) > 2.0:
        y_max = 3.0
    elif 2.0 > max(loss_max) > 1.0 or 2.0 > max(test_loss_max) > 1.0:
        y_max = 2.0
    else:
        y_max = 1.0

    ###

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(xLabel, fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_ylim(0.0, y_max)
    ax1.plot(x, loss_mean, 'r--', label='Training_Loss')
    ax1.plot(x, test_loss_mean, 'r-', label='Test_Loss')

    ax1.plot(x, test_loss_min, 'r-', alpha=0.01)
    ax1.plot(x, test_loss_max, 'r-', alpha=0.01)
    ax1.fill_between(x, test_loss_min, test_loss_max,
                     interpolate=True, color="red", alpha=0.1)

    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='upper right', prop={'size': 14})

    fig.suptitle(f"{title_str} - Loss - {str_trails}",
                 fontsize=14, fontweight="bold")
    ax1.set_title(subtitle, fontsize=13, color="gray")

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.savefig(path)  # "output/basic_impl_model_training_results.png")
    plt.close()
    plt.cla()


def run_logic(CASE: str, MODEL: str, TRAIN_INC_RANGE: range):
    csv_input = f"{MODEL}\\{CASE}_test_results.csv"
    path = "C:\\Users\\johns\\PycharmProjects\\VT\\" \
           "training_cv_models_on_ue_images_jws\\code\\" \
           "02_Classifier_Training\\output"

    df_csv = pd.read_csv(os.path.join(path, csv_input))

    for i, model in enumerate(list(df_csv['Model'].unique())):
        for j, case in enumerate(list(df_csv['Classes'].unique())):

            #for k in list(df_csv['id'].unique()):
            df_instance = df_csv.query(f'Model == \"{model}\" and '
                                        f'Classes == \"{case}\"')

            if df_instance.empty is False:
                # Average the results for each 'id'
                results_dict = {'id': [],
                                 'num': [],
                                 'accuracy': [],
                                 'test_Manual_accuracy': [],
                                 'loss': [],
                                 'test_loss': [],
                                 'Number of REAL Training Images': [],
                                 'Number of SYNTH Images': [],
                                 'Dataset Breakdown': []
                                }

                max_min_dict = {'id': [],
                                 'num': [],
                                 'acc_mean': [],
                                 'acc_min': [],
                                 'acc_max': [],
                                 'test_Manual_acc_mean': [],
                                 'test_Manual_acc_min': [],
                                 'test_Manual_acc_max': [],
                                 'loss_mean': [],
                                 'loss_min': [],
                                 'loss_max': [],
                                 'test_loss_mean': [],
                                 'test_loss_min': [],
                                 'test_loss_max': [],
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
                    results_dict['num'].append(len(df_r['accuracy']))
                    results_dict['accuracy'].append(df_r['accuracy'].mean())
                    results_dict['test_Manual_accuracy'].append(
                                    df_r['test_Manual_accuracy'].mean())
                    results_dict['loss'].append(df_r['loss'].mean())
                    results_dict['test_loss'].append(df_r['test_loss'].mean())

                    results_dict['Number of SYNTH Images'].append(
                            int(df_r['Number of SYNTH Images'].mean()))

                    breakdown = list(df_r['Dataset Breakdown (TRAIN/VAL/TEST)'])
                    results_dict['Dataset Breakdown'].append(breakdown)

                    breakdown = breakdown[0].split('[')[1].split(']')[0]
                    breakdown = breakdown.split(', ')
                    num_real_train = int(df_r['Number of REAL Images'].min()) \
                                    - float(breakdown[1]) - float(breakdown[2])
                    results_dict['Number of REAL Training Images'].append(
                                    int(num_real_train))

                    ######################################

                    max_min_dict['id'].append(int(id))
                    max_min_dict['num'].append(len(df_r['accuracy']))

                    max_min_dict['acc_mean'].append(df_r['accuracy'].mean())
                    max_min_dict['acc_min'].append(df_r['accuracy'].min())
                    max_min_dict['acc_max'].append(df_r['accuracy'].max())

                    max_min_dict['test_Manual_acc_mean'].append(
                        df_r['test_Manual_accuracy'].mean())
                    max_min_dict['test_Manual_acc_min'].append(
                        df_r['test_Manual_accuracy'].min())
                    max_min_dict['test_Manual_acc_max'].append(
                        df_r['test_Manual_accuracy'].max())

                    max_min_dict['loss_mean'].append(df_r['loss'].mean())
                    max_min_dict['loss_min'].append(df_r['loss'].min())
                    max_min_dict['loss_max'].append(df_r['loss'].max())

                    max_min_dict['test_loss_mean'].append(
                        df_r['test_loss'].mean())
                    max_min_dict['test_loss_min'].append(
                        df_r['test_loss'].min())
                    max_min_dict['test_loss_max'].append(
                        df_r['test_loss'].max())


                    max_min_dict['Number of SYNTH Images'].append(
                        int(df_r['Number of SYNTH Images'].mean()))

                    breakdown = list(df_r['Dataset Breakdown (TRAIN/VAL/TEST)'])
                    max_min_dict['Dataset Breakdown'].append(breakdown)

                    breakdown = breakdown[0].split('[')[1].split(']')[0]
                    breakdown = breakdown.split(', ')
                    num_real_train = int(df_r['Number of REAL Images'].min()) \
                                - float(breakdown[1]) - float(breakdown[2])
                    max_min_dict['Number of REAL Training Images'].append(
                                    int(num_real_train))


                df_avg = pd.DataFrame.from_dict(results_dict)
                df_avg = df_avg.sort_values(by=["id"])
                # Output Plots
                title = f"{MODEL} {CASE}"
                # output_acc_plot(df_avg, os.path.join(path,
                #                 f"{model}_{type}_{inc}_acc_plot.png"))
                #
                # output_loss_plot(df_avg, os.path.join(path,
                #                 f"{model}_{type}_{inc}_loss_plot.png"))

                df_trio = pd.DataFrame.from_dict(max_min_dict)
                df_trio = df_trio.sort_values(by=["id"])

                ###

                type = "cat_dog" if "cat" in case else "weld"
                inc = "IncSynth" if TRAIN_INC_RANGE[0] == 0 else "IncReal"

                if "cat" in case and "real" in inc.lower():
                    y_lim = (0.6, 1.0)
                elif "cat" in case and "synth" in inc.lower():
                    y_lim = (0.85, 1.0)
                elif "defect" in case.lower() and "real" in inc.lower():
                    y_lim = (0.4, 1.0)
                elif "defect" in case.lower() and "synth" in inc.lower():
                    y_lim = (0.7, 1.0)

                ###

                df_trio.to_csv(os.path.join(path, "_plots",
                                    f"{model}_{type}_{inc}_plots.csv"))

                p = os.path.join(path, "_plots",
                                f"{model}_{type}_{inc}_acc_plot.png")
                output_shadow_acc_plot(df_trio, title, p, y_range=y_lim)

                p = os.path.join(path, "_plots",
                                f"{model}_{type}_{inc}_loss_plot.png")
                output_shadow_loss_plot(df_trio, title, p)


if __name__=="__main__":
    CASE = ["cats_dogs", "weld"]
    MODEL = ["vgg16", "mobilenet"]
    INCREMENTS = [range(0,16), range(100,116)]

    # run_logic(CASE="weld",
    #           MODEL="vgg16",
    #           TRAIN_INC_RANGE=range(100,116))

    for model in MODEL:
        for case in CASE:
            for increment_type in INCREMENTS:
                run_logic(CASE=case, MODEL=model, TRAIN_INC_RANGE=increment_type)