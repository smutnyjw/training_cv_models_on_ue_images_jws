'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import os
from PIL import Image

base_path = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project" \
                 "\\09_Modify_UE_texture_maps\\"


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

WELD_TOL = 14
LINES_VERT = [  [305, 1332, 2357, 3382],
                [138, 1167, 2191, 3218],
                [651, 1674, 2699, 3723] ]
LINES_HORZ = [0, 1364, 2732, 4096]

#======================================================================

img_file = os.path.join(base_path, "parent_assets", "Weld_Edits",
                   f"T_Damaged_Welded_metal_normal.TGA")

img = Image.open(img_file)


for x in range(200):
    for y in range(2000):
        img[LINES_VERT[0][0] + x, 100 + y] = WHITE

output_file = os.path.join(base_path, "output", f"testw_normal.tga")
img.save(output_file, img)

