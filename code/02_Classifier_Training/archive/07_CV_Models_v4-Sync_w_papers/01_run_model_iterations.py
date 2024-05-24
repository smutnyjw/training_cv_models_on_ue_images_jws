'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import runpy
import os

import run_model

def run_vgg16():
    print("\n\n--------------\n\nRUN vgg16.py\n")
    runpy.run_path(os.path.join(os.getcwd(), 'vgg16.py'))

def run_mobilenet():
    print("\n\n--------------\n\nRUN mobileNetV3.py\n")
    runpy.run_path(os.path.join(os.getcwd(), 'mobileNetV3.py'))

###############################
path = os.path.join('output')

repeat = 3
cfgs = [0, 1, 2, 3, 4, 5, 6]

for j in [4]:
    run_model.main(type="VGG16", cfg_setting=j,
                           CASE_FLAG="Cats_Dogs",
                           base_path=path,
                           use_synth_dataset_list=[1, 1, 0, 0])

for j in [0, 2, 4]:
    run_model.main(type="VGG16", cfg_setting=j,
                           CASE_FLAG="Cats_Dogs",
                           base_path=path,
                           use_synth_dataset_list=[1, 1, 1, 1])


# for i in range(1):
#     for j in cfgs:
#         run_model.main(type="VGG16", cfg_setting=j,
#                        CASE_FLAG="Cats_Dogs",
#                        base_path=path,
#                        use_synth_dataset_list=[1, 1, 0, 0])
#         run_model.main(type="VGG16", cfg_setting=j,
#                        CASE_FLAG="Cats_Dogs",
#                        base_path=path,
#                        use_synth_dataset_list=[1, 1, 1, 1])

# # I think high dropout is and small Batch_size is killing it
# run_model.main(type="MobileNetV3", cfg_setting=0,
#                CASE_FLAG="Cats_Dogs",
#                base_path=path,
#                use_synth_dataset_list=[1, 1, 0, 0])
# run_model.main(type="MobileNetV3", cfg_setting=0,
#                CASE_FLAG="Cats_Dogs",
#                base_path=path,
#                use_synth_dataset_list=[1, 1, 0, 0])

# for i in range(3):
#     for j in cfgs:
#         run_model.main(type="VGG16", cfg_setting=j,
#                        CASE_FLAG="Cats_Dogs",
#                        base_path=path,
#                        use_synth_dataset_list=[1, 1, 1, 1])
#         run_model.main(type="MobileNetV3", cfg_setting=j,
#                        CASE_FLAG="Cats_Dogs",
#                        base_path=path,
#                        use_synth_dataset_list=[1, 1, 1, 1])
#
# for i in range(3):
#     for j in cfgs:
#         run_model.main(type="VGG16", cfg_setting=j,
#                        CASE_FLAG="Cats_Dogs",
#                        base_path=path,
#                        use_synth_dataset_list=[1, 1, 0, 0])
#         run_model.main(type="MobileNetV3", cfg_setting=j,
#                        CASE_FLAG="Cats_Dogs",
#                        base_path=path,
#                        use_synth_dataset_list=[1, 1, 0, 0])



# ###############################
#
# path = os.path.join('output')
# repeat = 3
# num_cfgs = 5
#
# t0_list = []
# t1_list = []
# t2_list = []
# t3_list = []
# t4_list = []
# t5_list = []
# output = "\n\n------- Results of the output experiments -------\n\n"
#
# for i in range(1, num_cfgs, 1):
#     for j in range(repeat):
#         # Test_manual_acc, test_binary_acc, train_acc
#         t0, t1, t2 = run_model.main(type="VGG16", cfg_setting=i,
#                                     base_path=path)
#         t0_list.append(round(t0, 4))
#         t1_list.append(round(t1, 4))
#         t2_list.append(round(t2, 4))
#
#         # Test_manual_acc, test_binary_acc, train_acc
#         t3, t4, t5 = run_model.main(type="MobileNetV3", cfg_setting=i,
#                                     base_path=path)
#         t3_list.append(round(t3, 4))
#         t4_list.append(round(t4, 4))
#         t5_list.append(round(t5, 4))
#
#     output += f"*** cfg {i}\n"
#     output += "* avg/max/min\n"
#     output += f"VGG16:\n" \
#             f"test_m_acc: {sum(t0_list)/len(t0_list)} / " \
#                             f"{max(t0_list)} / {min(t0_list)}\n" \
#             f"test_b_acc: {sum(t1_list)/len(t1_list)}/ " \
#                             f"{max(t1_list)} / {min(t1_list)}\n" \
#             f"train_acc:  {sum(t2_list)/len(t2_list)} / " \
#                             f"{max(t2_list)} / {min(t2_list)}\n\n"
#     output += f"mobilenet:\n" \
#             f"test_m_acc: {sum(t3_list) / len(t3_list)} / " \
#                             f"{max(t3_list)} / {min(t3_list)}\n" \
#             f"test_b_acc: {sum(t4_list) / len(t4_list)}/ " \
#                             f"{max(t4_list)} / {min(t4_list)}\n" \
#             f"train_acc: {sum(t5_list) / len(t5_list)} / " \
#                             f"{max(t5_list)} / {min(t5_list)}\n\n"
#
#     output += "*** Raw\n"   \
#         f"VGG16:\n" \
#         f"test_m_acc: {t0_list}\n" \
#         f"test_b_acc: {t1_list}\n" \
#         f"train_acc:  {t2_list}\n\n" \
#         f"MobileNet:\n" \
#         f"test_m_acc: {t3_list}\n" \
#         f"test_b_acc: {t4_list}\n" \
#         f"train_acc:  {t5_list}\n"
#
# print(output)

# run_vgg16()
# run_mobilenet()


