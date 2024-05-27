/02_Classifier_Training directory

This directory contains the python scripts required to train and test the binary
 classifier models. Run the 'main.py' python script to train the specific model
 configuration(s). The 'main.py' file will call on /run_model.py to perform
 model training and testing.


The user should edit the /main.py file and
/run_model::run_a_specific_setting() method to train the models on the exact
amount of synthetic and real data that the user desires. The user should edit
the /cfg_vgg16.py and /cfg_mobilenet.py files to edit the file paths to datasets
 and other training settings as desired.
