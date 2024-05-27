'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import os
import cv2
import lib_cmn as cmn

import random


base_path = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project"

input_img_path = os.path.join(base_path, "materials", "parent_assets")
output_img_path = os.path.join(base_path, "09_Modify_UE_texture_maps", "output")
simple_maps = [
                "T_Welded_metal_normal.TGA.bmp",
                "T_Welded_metal_ambientocclusion.bmp",
                "T_Welded_metal_metallic.bmp",
                "T_Welded_metal_roughness.bmp"
            ]

color_change = [cmn.BLACK,# +[0],
                cmn.BLACK,
                cmn.LGRAY,
                cmn.WHITE] #[200, 200, 200]] #cmn.WHITE]

#======================================================================

# OUTPUT BITMAP IMAGES

random_seed = random.randint(1, 1000)

for i in range(len(simple_maps)):
    ue_map = simple_maps[i]
    color = color_change[i]
    ###
    # Generate one crack for one single map
    img = cv2.imread(os.path.join(input_img_path, ue_map),
                     cv2.IMREAD_UNCHANGED)

    if color == 'skip':
        cv2.imwrite(os.path.join(output_img_path, ue_map + ".crack.bmp"),
                    img)
        continue


    #
    # Add Crack
    mod_img = cmn.add_crack(center_x=cmn.LINES_VERT[1][0],
                            center_y=cmn.LINES_HORZ[2],
                            length=100,
                            segments=10,
                            tol=int(cmn.WELD_TOL)-3,
                            thickness=2,
                            rotation=90,
                            defect_flag=True,
                            color=color,
                            img_data=img,
                            seed=random_seed)

    #
    # Add a Defect Crack
    mod_img = cmn.add_crack(center_x=cmn.LINES_VERT[1][0],
                            center_y=cmn.LINES_HORZ[1],
                            length=100,
                            segments=10,
                            tol=int(cmn.WELD_TOL)-3,
                            thickness=3,
                            rotation=270,
                            defect_flag=True,
                            color=color,
                            img_data=mod_img,
                            seed=random_seed)

    ###
    # Output the new map
    cv2.imwrite(os.path.join(output_img_path, ue_map +".crack.bmp"),
                    mod_img)
