'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import unreal


############################
#   Unreal Assets

home_path = '/Game/MENG_POC_04_ImagePipeline01/TextureSelect/'
MATERIAL_PATHS = [
    home_path + 'background_select_Mat_Inst.background_select_Mat_Inst',
    home_path + 'scene_select_Mat_Inst.scene_select_Mat_Inst',
    home_path + 'floor_select_Mat_Inst.floor_select_Mat_Inst'
    ]

SCENE_PRESETS = {
    # Multiplexer decoding of material instance parameters to select a texture.
    #   Switch values are based on LERP nodes, not switches.
    #   Values must be 0.0 or 1.0
    "forest": [0, 0, 0],
    "grassland": [1, 0, 0],
    "urban": [0, 0, 1],
    "interior": [0, 1, 1]
}
MATERIAL_PARAMS = ['sw_1a', 'sw_1b', 'sw_2']

# Checks
editor = unreal.MaterialEditingLibrary()
for i in range(len(MATERIAL_PATHS)):
    obj = unreal.load_asset(None, MATERIAL_PATHS[i])
    assert obj

    if len(obj.get_scalar_parameter_names()) != len(MATERIAL_PARAMS):
        num_uasset = len(obj.get_scalar_parameter_names())
        num_py = len(MATERIAL_PARAMS)
        raise Exception("*** ERROR: Double check specified Material "
                        "Instances. There are too many material parameters "
                        f"listed in the python script {num_uasset} compared to the ones "
                        f"available in the Material_Inst uasset file {num_py}.")

for i in range(len(SCENE_PRESETS)):
    switches = SCENE_PRESETS[i]
    if len(switches) != len(MATERIAL_PARAMS):
        raise Exception("*** Error: update PRESET settings. Unequal "
                        f"dimension for setting index {i}")


############################
#   Camera settings

model_locations = {
    "CENTER_POINT": [-1, -1, -1],
    "START": [-1, -1, -1],
    "END": [-1, -1, -1]
}

cam_home = [0, -300, 200]
cam_offset = [0.0, 0.0, 10.0]

range_rho = [-350]  #[-200, -350]
range_alpha = [0, 90, 180, 270]#  [90, 180, 270]
range_phi = [45, 90]



