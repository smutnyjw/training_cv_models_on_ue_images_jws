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


base_path = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project"

input_img_path = os.path.join(base_path, "materials", "parent_assets")
output_img_path = os.path.join(base_path, "09_Modify_UE_texture_maps", "output")

# TODO - 1) Implment the map+color as a dictionary, 2) have all map changes
#  occur in a single function call.
simple_maps = [
                "T_Welded_metal_normal.TGA.bmp",
                "T_Welded_metal_ambientocclusion.bmp",
                "T_Welded_metal_metallic.bmp",
                "T_Welded_metal_roughness.bmp"
            ]

color_change = [[-1, -1, -1],
                cmn.BLACK,
                cmn.LGRAY,
                cmn.DGRAY] #[200, 200, 200]] #cmn.WHITE]

#======================================================================

# OUTPUT BITMAP IMAGES

for i in range(len(simple_maps)):
    ue_map = simple_maps[i]
    color = color_change[i]
    ###
    # Generate one Porosity oval for one single map
    img = cv2.imread(os.path.join(input_img_path, ue_map),
                     cv2.IMREAD_UNCHANGED)

    if color == 'skip':
        cv2.imwrite(os.path.join(output_img_path, ue_map + ".porosity.bmp"),
                    img)
        continue

    #
    # Add Porosity holes
    offset = -2
    mod_img = cmn.add_oval(center_x=cmn.LINES_VERT[0][0] + offset,
                           center_y=390,
                           radius_x=int(cmn.WELD_TOL*0.5)-2,    # TODO - /2 ???
                           radius_y=20,
                           defect_flag=True,
                           img_data=img, color=color, seed=5)

    offset = -2
    mod_img = cmn.add_oval(center_x=cmn.LINES_VERT[0][0] + offset,
                           center_y=320,
                           radius_x=int(cmn.WELD_TOL * 0.5) - 3,
                           radius_y=15,
                           defect_flag=True,
                           img_data=mod_img, color=color, seed=5)

    offset = 0
    mod_img = cmn.add_oval(center_x=cmn.LINES_VERT[0][0] + offset,
                           center_y=350,
                           radius_x=int(cmn.WELD_TOL * 0.5) - 3,
                           radius_y=6,
                           defect_flag=False,
                           img_data=mod_img, color=color, seed=5)

    offset = 3
    mod_img = cmn.add_oval(center_x=cmn.LINES_VERT[0][0] + offset,
                           center_y=410,
                           radius_x=int(cmn.WELD_TOL * 0.5) - 1,
                           radius_y=10,
                           defect_flag=True,
                           img_data=mod_img, color=color, seed=5)

    ###
    # Output the new map
    cv2.imwrite(os.path.join(output_img_path, ue_map +".porosity.bmp"),
                    mod_img)
