'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import os
import pandas as pd
from copy import deepcopy

test_dir = "weld-ProportionalWeighting-ForceSample"
model = "mobilenet" if 1 else "vgg16"
dir_to_results = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project" \
                 "\\08_CV_Models_v5-After_Meeting_Sync\\output" \
                 f"\\{model}\\Weld\\{test_dir}"
output_file = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project" \
                 f"\\08_CV_Models_v5-After_Meeting_Sync\\output" \
              f"\\{model}\\Weld\\{test_dir}.csv"

# dir_to_results = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project" \
#                  "\\08_CV_Models_v5-After_Meeting_Sync\\output" \
#                  f"\\{model}\\pending_review"

_quantities_empty = {
    "id": -1,
    "Model": "str",
    "Classes": "str",
    "Number of Images": -1,
    "Number of REAL Images": -1,
    "Number of SYNTH Images": -1,
    "Number of SYNTH Images available": -1,
    "Real_Synth ratio - TRAIN": [],
    "Dataset Breakdown (TRAIN/VAL/TEST)": [],
    "DatasetSplit": [],
    "Class Distribution": [-1, -1],
    "Class Distribution - Real Images": [-1, -1],
    "Class Distribution - Train Images": [-1, -1],
    "Class Distribution - Val Images": [-1, -1],
    "Class Distribution - Test Images": [-1, -1],
    "Class Weights": [1, 1],
    "Total Epochs Trained": -1,
    "test_cost_per_sample": -1,
    "accuracy": -1,
    "test_Manual_accuracy": -1,
    "loss": -1,
    "test_loss": -1,
    "auc": -1,
    "test_auc": -1,
    "test_Manual_TP": -1,
    "test_Manual_TN": -1,
    "test_Manual_FP": -1,
    "test_Manual_FN": -1,
    "test_accuracy": -1,
    "test_recall": -1,
    "test_precision": -1,
    "recall": -1,
    "precision": -1,
}

extracted_data = []
for dir in os.listdir(dir_to_results):
    abs_path = os.path.join(dir_to_results, dir)

    for file in os.listdir(abs_path):
        if "REF_run_info" in file:
            _quantities = deepcopy(_quantities_empty)
            _quantities['id'] = abs_path[-1:]
            abs_file = os.path.join(abs_path, file)
            f = open(abs_file)
            for line in f:
                for key in list(_quantities.keys()):
                    if line.split(":")[0] == key:
                        _quantities[key] = line.split("\t")[1].split("\n")[0]
                        break
            extracted_data.append(list(_quantities.values()))


df_data = pd.DataFrame(extracted_data, columns=list(_quantities.keys()))
df_data = df_data.sort_values(by=["id"])

df_data.to_csv(output_file)
