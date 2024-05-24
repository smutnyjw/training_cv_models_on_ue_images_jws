'''
File:   01_run_model_iterations.py
Author: John Smutny
Date:   03/26/2024
Description: 
    A base file that will enable GPUs used during the calling of a separate
    python script/method for model training.

    This file defines which configurations are run and how many times. The
    user can set which model {vgg16, mobilenet} is trained and for which use
    case.

    The acceptable configurations are set by the user in ...
        run_model::run_a_specific_setting()

Other Notes:

'''


import os
import gc
import tensorflow as tf

import run_model

###############################

def set_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'
    os.environ['TF_RUN_EAGER_OP_AS_FUNCTION'] = 'false'


if __name__ == "__main__":
    # Key
    path = os.path.join('output')
    cfgs = [0, 1, 2, 3, 4, 5, 6]
    r_cfgs = [6, 5, 4, 3, 2, 1, 0]
    experiment = [11, 12, 13, 14, 15]

    weld_base = [20]
    weld_experiment01 = [21, 22, 23]
    weld_experiment02 = [24, 25, 26, 27]

    ###
    # Attempt 1)
    #   This is my main way of running my models. Call an isolated ::main()
    #   method that will execute based on input parameters I give over the method
    #   parameters.
    #
    #   valid values for 'type' input: {VGG16, mobilenetv3}

    for i in range(0):
        for j in range(20, 28):
            set_gpus()
            run_model.run_experiment(type="VGG16", cfg_setting=j,
                                        CASE_FLAG="Weld",
                                        base_path=path,
                                        use_synth_dataset_list=[1])
            gc.collect()

    for i in range(1):
        for j in range(20, 28):
            set_gpus()
            run_model.run_experiment(type="mobilenetv3", cfg_setting=j,
                                        CASE_FLAG="Weld",
                                        base_path=path,
                                        use_synth_dataset_list=[1])
            gc.collect()



    ##########################################################################
    ###
    # Attempt 2)
    #   Use a method that would allow me to execute a python script via a python
    #   script. However, the documentation for the script does not seem to
    #   support input parameters


    # import runpy
    # runpy.run_path(os.path.join(os.getcwd(), 'run_model.py'))

    ###
    # Attempt 3)
    #   Use a system command to start the python script, however this scrip is
    #   not run in the same namespace as my PyCharm execution of the same .py
    #   file. It is possible for me to install all the necessary packages,
    #   but I am reluctant to do this for fear of breaking my conda system (not
    #   just my virtual environment).
    # os.system('python3 run_model.py -h')
