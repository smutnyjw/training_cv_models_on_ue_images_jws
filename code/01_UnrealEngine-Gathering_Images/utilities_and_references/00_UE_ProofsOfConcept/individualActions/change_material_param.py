'''
File:       change_material_param
Author: 
Date:   
Description: 
    First attempt to interface with a

Other Notes:

'''
import random
import time
import unreal

def toggle_param(material_path, param):
    ############################
    # Get the material instance
    #
    # Lessons:
    # 1) Cannot use the 'EditorAssetLibrary()' class. must be
    # 'MaterialEditingLibrary()'
    # 2) Must use get the material instance as a 'object' rather than 'asset'
    blue_mat = unreal.load_object(None, material_path)
    obj = blue_mat
    assert obj

    print(obj)

    # Instance Details - Class = MaterialInstanceConstant()
    mat_name = obj.get_name()
    mat_parent = obj.get_class()
    mat_metal = obj.get_scalar_parameter_value(param)

    print(f"Details:\n{mat_name}\n{mat_parent}\n{mat_metal}")

    print(obj.scalar_parameter_values)

    #################################

    # Toggle "rough" parameter using MaterialEditingLibrary() methods
    editor = unreal.MaterialEditingLibrary()
    val = editor.get_material_instance_scalar_parameter_value(obj, param)
    val = 0 if val else 1

    x = editor.set_material_instance_scalar_parameter_value(obj, param, val)
    print(x)
    print(obj.scalar_parameter_values)
    print(editor.get_material_instance_scalar_parameter_value(obj, param))
    # editor.recompile_material(obj)
    # print(editor.get_material_instance_scalar_parameter_value(obj, "rough"))


def set_param(mat, param: str, val: float):
    obj = unreal.load_object(None, mat)
    assert obj

    print(obj)
    print(obj.scalar_parameter_values)
    # Set material parameter using MaterialEditingLibrary() methods
    editor = unreal.MaterialEditingLibrary()
    # c_val = editor.get_material_instance_scalar_parameter_value(mat, param)
    editor.set_material_instance_scalar_parameter_value(obj, param, val)


####
# Main

for x in [0]:
    material_path = '/Game/_MENG_Project/00_Proof_of_Concept/metal_p_blue1.metal_p_blue1'
    param = "rough"
    # toggle_param(material_path, param)


    material_path = '/Game/_MENG_Project/00_Proof_of_Concept/metal_p_rust1_Inst1' \
                    '.metal_p_rust1_Inst1'
    # param = "text1_rough"
    # toggle_param(material_path, param)
    #
    # param = "text2_rough"
    # toggle_param(material_path, param)

    material = '/Game/_MENG_Project/00_Proof_of_Concept/'\
               'metal_p_rust_alpha_Inst1.metal_p_rust_alpha_Inst1'
    param = "alpha_x"
    val = random.randrange(-70.0, 40, 1) / 100
    set_param(material, param, val)

    param = "alpha_y"
    val = random.randrange(-70, 50, 1) / 100
    set_param(material, param, val)



