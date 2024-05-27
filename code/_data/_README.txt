/_data Directory

By default, this directory is directly referenced by /00_ImagePipeline scripts
and /02_Classifier_Training/ in order to output .jpg screenshots (and associated
 .csv list) and then use those images for model training.

This is a place to organize your data. It is not required. Please update any
paths with your changes (such as /02_Classifier_Training/ cfg_vgg16.py and
cfg_mobilenet.py).