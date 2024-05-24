'''
File:       change_actor_texture.py
Author:     John Smutny
Date:       02/22/2024
Description: 
    Successful cycling of textures for objects in an UE4 environment. The
    functionality is dependent on a user creating a Material that is
    compatible with these actions.

    Requirements for the Material file:
    1. Create individual TextureSample nodes specifying which textures you
        want to be present in the object.
    2. Connect each TextureSample to LERP nodes with a SingleParam to be
        controlled


    Requirements for the Unreal Editor files
    1. Every object that should be separately be controlled must be textured
        with a separate Material Instance file.


Other Notes:

'''
import random

import unreal




#######################################################
#           Test backgrounds - First attempt (background, floor, scene)
#######################################################

home_path = '/Game/MENG_POC_04_ImagePipeline01/TextureSelect/'
material_paths = [
    home_path + 'background_select_Mat_Inst.background_select_Mat_Inst',
    home_path + 'scene_select_Mat_Inst.scene_select_Mat_Inst',
    home_path + 'floor_select_Mat_Inst.floor_select_Mat_Inst'
    ]

MATERIAL_PARAMS = ['sw_1a', 'sw_1b', 'sw_2']

VALID_OPTIONS = ["A", "B", "C", "D"]

def decode_settings(settings: list):
    for i in range(len(settings)):
        if settings[i] == "A":
            settings[i] = [0.0, 0.0, 0.0]
        elif settings[i] == "B":
            settings[i] = [1.0, 0.0, 0.0]
        elif settings[i] == "C":
            settings[i] = [0.0, 0.0, 1.0]
        elif settings[i] == "D":
            settings[i] = [0.0, 1.0, 1.0]

    return settings


#######################################################
#           Start of MAIN()
#######################################################

texture_paths = []

#####
# Code to print out all or specific actors in a loaded level    - Confirmed
editor = unreal.EditorLevelLibrary()
actors = editor.get_all_level_actors()

for y in actors:
    if y.get_name() == 'Sphere_Brush_StaticMesh':
        sphere = y
        print("*** DEBUG: Found Sphere object")


    if y.get_name() == 'Floor':
        floor = y
        print("*** DEBUG: Found Floor object")


def set_param(mat, param: str, val: float):
    obj = unreal.load_object(None, mat)
    assert obj
    #print(obj.scalar_parameter_values)

    # Set material parameter using MaterialEditingLibrary() methods
    editor = unreal.MaterialEditingLibrary()

    # c_val = editor.get_material_instance_scalar_parameter_value(mat, param)
    editor.set_material_instance_scalar_parameter_value(obj, param, val)


random_settings = random.choices(VALID_OPTIONS, k=4)
settings = decode_settings(random_settings)
for i in range(len(material_paths)):
    mp = material_paths[i]
    print(f"*** DEBUG: Attempt to set {mp}")

    for j in range(len(MATERIAL_PARAMS)):
        set_param(mp, MATERIAL_PARAMS[j], settings[i][j])




# #######################################################
# #           Test backgrounds - Letters
# #######################################################
#
# home_path = '/Game/MENG_POC_04_ImagePipeline01/TextureSelect' \
#             '/zTestTextureSelect/'
# material_paths = [
#     home_path + 'letter_select_Mat_Inst01.letter_select_Mat_Inst01',
#     home_path + 'letter_select_Mat_Inst02.letter_select_Mat_Inst02',
#     home_path + 'letter_select_Mat_Inst03.letter_select_Mat_Inst03',
#     home_path + 'letter_select_Mat_Inst04.letter_select_Mat_Inst04'
#     ]
#
# MATERIAL_PARAMS = ['Param_AB', 'Param_CD', 'Param_sel2']
#
# VALID_LETTERS = ["A", "B", "C", "D"]
#
# def decode_letters(settings: list):
#     for i in range(len(settings)):
#         if settings[i] == "A":
#             settings[i] = [0.0, 0.0, 0.0]
#         elif settings[i] == "B":
#             settings[i] = [1.0, 0.0, 0.0]
#         elif settings[i] == "C":
#             settings[i] = [0.0, 0.0, 1.0]
#         elif settings[i] == "D":
#             settings[i] = [0.0, 1.0, 1.0]
#
#     return settings
#
# #######################################################
# #           Start of MAIN()
#
# texture_paths = []
#
# #####
# # Code to print out all or specific actors in a loaded level    - Confirmed
# editor = unreal.EditorLevelLibrary()
# actors = editor.get_all_level_actors()
#
# for y in actors:
#     #print(f"Actor: {y.get_name()}")
#
#     if 'bg' in y.get_name():
#         if 'left' in y.get_name():
#             bg_left = y
#             print("*** DEBUG: Found background_wall_LEFT")
#
#         elif 'center' in y.get_name():
#             bg_center = y
#             print("*** DEBUG: Found background_wall_CENTER")
#
#         elif 'right' in y.get_name():
#             bg_right = y
#             print("*** DEBUG: Found background_wall_RIGHT")
#
#         else:
#             raise Exception("*** ERROR: Double check Wall names. "
#                             f"{y.get_name} not registered.")
#
#     if y.get_name() == 'Floor':
#         floor = y
#         print("*** DEBUG: Found Floor object")
#
#
# def set_param(mat, param: str, val: float):
#     obj = unreal.load_object(None, mat)
#     assert obj
#     #print(obj.scalar_parameter_values)
#
#     # Set material parameter using MaterialEditingLibrary() methods
#     editor = unreal.MaterialEditingLibrary()
#
#     # c_val = editor.get_material_instance_scalar_parameter_value(mat, param)
#     editor.set_material_instance_scalar_parameter_value(obj, param, val)
#
#
# random_letters = random.choices(VALID_LETTERS, k=4)
# letters = decode_letters(random_letters)
# for i in range(len(material_paths)):
#     mp = material_paths[i]
#     print(f"*** DEBUG: Attempt to set {mp}")
#
#     for j in range(len(MATERIAL_PARAMS)):
#         set_param(mp, MATERIAL_PARAMS[j], letters[i][j])
