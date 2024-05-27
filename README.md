# training_cv_models_on_ue_images_jws
This repository contains documents and source code related to John W. Belanger Smutny's Viriginia Tech Master's of Engineering senior capstone project on "The Effect of Training Published Computer Vision Models on Unreal Engine Synthetic Images". An Introduction to the Synergy between Graphics Rendering Software and Machine Learning.


## Acknowledgements
Thank you to Dr Creed Jones, Dr Ryan Williams, and Dr Jason Xuan for being a part of my committee, especially to Dr Jones for giving me their ear and guidance throughout this project.

Additional thank yous to …

Northrop Grumman Corporation for funding my Master’s Degree at Virginia Tech University.

To the numerous teachers that I consulted with throughout this project. To Brandon Padayao with Baltimore OpenWorks for allowing me to sit in on a welding course and learn from you about metal welding. To Riley Van Etten and Deion Waddell for their expertise in 3D graphic design. To numerous others helping me brainstorm through ideas and barriers.


This project gave me the chance to learn about one of the most powerful graphics engines - one of many that fueled my imagination as a child.


# Updates

## (update - May 27th, 2024 - Source Code is available)
In the latest round of commits, the source code used to gather all synthetic
images and to train+test the computer vision classifiers is available for
download and use. Please see all README.txt files in each subsequent director
for a brief background on the contents of each directory. The README.txt files
and .py script headers highlight specific variables that you should change (like
Input/Output paths) in order to operate the scripts effectively.

### /code/01_UnrealEngine-Gathering_Images
    This directory contains scripts run through your Unreal Engine Editor
    environment through the UE Python API. In order to use the scripts EXACTLY
    as written, you must setup a UE Environment with identical names and object
    types. It is highly recommended that you use all scripts as a base for you
    to customize for your own use.

    This directory also contains the base 'class.py' file used to execute
    specific UE Commands from tick callback.

### /code/02_Classifier_Training
    This directory contains all scripts related to training and testing the
    computer vision models. Please update all input and output paths in the
    related 'cfg_vgg16.py' and 'cfg_mobilenet.py' files.


## (update - May 16th, 2024 - No source code yet)
Currently there is no publicly available source code. The author is spending the rest of May to clean the source code, provide sufficient comments, and understanding for viewers. Please come back Monday June 10th, 2024 for another update. In the meantime, the research paper and engineering design document pdfs are available to view and download. Thank you for your patience. 




