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

color_change = [cmn.norm_SURFACE,
                cmn.GRAY,
                cmn.metallic_BG,
                cmn.rough_BG]

#======================================================================

# OUTPUT BITMAP IMAGES

random_seed = random.randint(1, 1000)

for i in range(len(simple_maps)):
    ue_map = simple_maps[i]
    color = color_change[i]

    # Generate one Incomplete Weld shape for one single map
    img = cv2.imread(os.path.join(input_img_path, ue_map),
                     cv2.IMREAD_UNCHANGED)

    if color == 'skip':
        cv2.imwrite(os.path.join(output_img_path, ue_map + ".incomplete.bmp"),
                    img)
        continue

    # Good max_radius range: defect = 20, non-defect < 10
    mod_img = cmn.add_incomplete_weld(center_x=cmn.LINES_VERT[0][0],
                                      center_y=400,
                                      cut_dir="r",
                                      max_radius=20,
                                      defect_flag=True,
                                      img_data=img,
                                      color=color,
                                      seed=20
                                      )

    mod_img = cmn.add_incomplete_weld(center_x=cmn.LINES_VERT[0][1],
                                      center_y=400,
                                      cut_dir="L",
                                      max_radius=20,
                                      defect_flag=True,
                                      img_data=mod_img,
                                      color=color,
                                      seed=20
                                      )

    mod_img = cmn.add_incomplete_weld(center_x=cmn.LINES_VERT[0][0],
                                      center_y=600,
                                      cut_dir="r",
                                      max_radius=8,
                                      defect_flag=False,
                                      img_data=mod_img,
                                      color=color,
                                      seed=20
                                      )

    mod_img = cmn.add_incomplete_weld(center_x=cmn.LINES_VERT[0][1],
                                      center_y=600,
                                      cut_dir="L",
                                      max_radius=8,
                                      defect_flag=False,
                                      img_data=mod_img,
                                      color=color,
                                      seed=20
                                      )

    mod_img = cmn.add_incomplete_weld(center_x=550,
                                      center_y=cmn.LINES_HORZ[1],
                                      cut_dir="u",
                                      max_radius=20,
                                      defect_flag=True,
                                      img_data=mod_img,
                                      color=color,
                                      seed=20
                                      )

    mod_img = cmn.add_incomplete_weld(center_x=650,
                                      center_y=cmn.LINES_HORZ[1],
                                      cut_dir="d",
                                      max_radius=8,
                                      defect_flag=False,
                                      img_data=mod_img,
                                      color=color,
                                      seed=20
                                      )


    ###
    # Output the new map
    cv2.imwrite(os.path.join(output_img_path, ue_map + ".incomplete.bmp"),
                mod_img)