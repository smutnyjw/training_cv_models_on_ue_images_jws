'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import os
import cv2

base_path = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project" \
                 "\\09_Modify_UE_texture_maps\\"


RED = [0, 0, 255]
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]

WELD_TOL = 14
LINES_VERT = [  [310, 1332, 2357, 3382],
                [138, 1167, 2191, 3218],
                [651, 1674, 2699, 3723] ]
LINES_HORZ = [0, 1364, 2732, 4096]

#======================================================================

test = [2]

for x in test:
    if x == 1:
        type = "ambientocclusion"
        file = f"T_Welded_metal_{type}.BMP"
    elif x == 2:
        type = "normal"
        # file = f"T_Welded_metal_{file}.BMP"
        file = f"T_Welded_metal_normal.TGA.BMP"
    elif x == 3:
        type = "metallic"
        file = f"T_Welded_metal_{type}.BMP"
    elif x == 4:
        type = "roughness"
        file = f"T_Welded_metal_{type}.BMP"

    image_file = os.path.join(base_path, "parent_assets", "Weld_Edits", file)

    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

    for x in range(200):
        for y in range(2000):
            if len(img[0][0]) == 3:
                img[100+x, LINES_VERT[0][0]+y] = WHITE
                # img[LINES_VERT[0][0]+x, 100+y] = WHITE
            elif len(img[0][0]) == 4:
                img[100+x, LINES_VERT[0][0]+y] = WHITE + [255]
                # img[LINES_VERT[0][0]+x, 100+y] = WHITE + [255]

    cv2.imwrite(os.path.join(base_path, "output", f"testw_{type}.bmp"), img)


    #cv2.imshow("test", img)
