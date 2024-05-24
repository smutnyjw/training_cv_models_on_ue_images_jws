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


base_path = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project" \
                 "\\09_Modify_UE_texture_maps\\"

input_img_path = os.path.join("parent_assets", "Weld_Edits")

# TODO - 1) Implment the map+color as a dictionary, 2) have all map changes
#  occur in a single function call.
simple_maps = [
                "T_Welded_metal_normal.TGA.bmp",
                "T_Welded_metal_ambientocclusion.bmp",
                "T_Welded_metal_metallic.bmp",
                "T_Welded_metal_roughness.bmp"
            ]

color_change = [cmn.WHITE,
                cmn.WHITE,
                cmn.BLACK,
                cmn.WHITE] #[200, 200, 200]] #cmn.WHITE]

output_img_path = os.path.join(base_path, "output")


#======================================================================

# OUTPUT BITMAP IMAGES


for i in range(len(simple_maps)):
    ue_map = simple_maps[i]
    color = color_change[i]
    ###
    # Generate one Porosity oval for one single map
    img = cv2.imread(os.path.join(base_path, input_img_path, ue_map),
                     cv2.IMREAD_UNCHANGED)

    for j in range(len(cmn.LINES_VERT[0])):
        img = cmn.add_oval(center_x=cmn.LINES_VERT[0][j],
                           center_y=cmn.LINES_HORZ[1],
                           radius_x=3,
                           radius_y=3,
                           img_data=img, color=color)

    for j in range(len(cmn.LINES_VERT[1])):
        for k in range(1, len(cmn.LINES_HORZ)-1):
            img = cmn.add_oval(center_x=cmn.LINES_VERT[1][j],
                                   center_y=cmn.LINES_HORZ[k],
                                   radius_x=3,
                                   radius_y=3,
                                   img_data=img, color=color)

    for j in range(len(cmn.LINES_VERT[2])):
        img = cmn.add_oval(center_x=cmn.LINES_VERT[2][j],
                           center_y=cmn.LINES_HORZ[2],
                           radius_x=3,
                           radius_y=3,
                           img_data=img, color=color)

    ###
    # Output the new map
    cv2.imwrite(os.path.join(output_img_path, ue_map +".reference.bmp"),
                    img)