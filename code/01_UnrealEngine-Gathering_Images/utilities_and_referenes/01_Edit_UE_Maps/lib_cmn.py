'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''
import math
import random
from math import sin, cos

import numpy as np

RED = [0, 0, 255]
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]

DGRAY = [87, 87, 87]
GRAY = [150, 150, 150]
LGRAY = [206, 206, 206]

norm_PURPLE = [233, 69, 84]
norm_BLUE = [220, 158, 44]
norm_LBLUE = [220, 203, 83]
norm_PINK = [242, 129, 189]
norm_SURFACE = [255, 126, 127]
rough_BG = [51, 51, 51]
metallic_BG = WHITE
ao_BG = [254, 254, 254]

GRADIENT = [-1, -1, -1]

WELD_TOL = 14
LINES_VERT = [[308, 1333, 2358, 3382],  # Intersection of x-axis
              [141, 1169, 2193, 3218],
              [651, 1675, 2699, 3723]]
LINES_HORZ = [0, 1365, 2732, 4095]      # Intersection of y-axis


def add_random_noise(val: list, noise_factor: int):
    random.seed(None)

    new_val = [0]*len(val)

    for i in range(len(val)):
        new_val[i] = val[i] + random.randint(-noise_factor, noise_factor)

    return new_val

def add_oval(center_x: int, center_y: int,
             radius_x: int, radius_y: int,
             defect_flag: int, img_data, color, seed):
    """
  Adds an oval shape to an already established data array

  Args:
    center_x: The x-coordinate of the center of the oval.
    center_y: The y-coordinate of the center of the oval.
    radius_x: The x-radius of the oval.
    radius_y: The y-radius of the oval.

  Returns:
    A numpy array representing the oval pixel image.
  """

    mod_img_data = img_data.copy()
    random.seed(seed)

    # Overwrite settings if this modification is not considered a defect
    if defect_flag == False:
        radius_x = random.randint(1, 3)
        radius_y = random.randint(1, 3)

    # Iterate over the pixels in the image.
    for y in range(center_y - radius_y, center_y + radius_y):
        for x in range(center_x - radius_x, center_x + radius_x):
            ## Calculate the distance from the pixel to the center of the oval.
            # distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            # If the distance is less than or equal to the radius, then the pixel
            # is inside the oval.
            # if distance <= radius_x and distance <= radius_y:

            x_p = x - center_x
            y_p = y - center_y

            if (x_p ** 2 / radius_x ** 2 + y_p ** 2 / radius_y ** 2) <= 1:
                # Determine the pixel's color
                if color == GRADIENT:
                    if x < center_x:
                        p_color = add_random_noise(norm_PINK, 20)
                    elif x == center_x:
                        p_color = add_random_noise(norm_LBLUE, 20)
                    elif x > center_x:
                        p_color = add_random_noise(norm_BLUE, 20)
                else:
                    p_color = color

                # Set the pixel's color
                if len(mod_img_data[0][0]) == 4:
                    p_color = p_color + [255]

                mod_img_data[y, x] = p_color

    # Return the image.
    return mod_img_data


def add_crack(center_x: int, center_y: int, length: int,
              segments: int, tol: int, thickness: int, rotation: int,
              defect_flag: int, color, img_data, seed):
    mod_img_data = img_data.copy()

    if defect_flag == False:
        tol = 1
        min_thick = 0
        max_thick = 1
    else:
        min_thick = 1
        max_thick = thickness

    ###
    # Set up the random parameters associated with making a crack
    random.seed(seed)

    noise = int(length*0.1/segments)
    len_offsets = [random.randint(-noise, noise) for i in range(segments+1)]

    # Define the line
    seg_pts = [int(length/segments)+len_offsets[i] for i in range(segments+1)]
    seg_pts = [seg_pts[i] + sum(seg_pts[:i]) for i in range(1, segments+1)]
    seg_pts[-1] = length
    seg_hgts = [random.randint(-tol, tol) for i in range(segments)]
    seg_thickness = [random.randint(min_thick, max_thick) for i in range(segments)]

    ###
    # Create the crack lines
    n_color = add_random_noise(color, 10) if len(color) == 4 else color

    crack_pts = []
    for i in range(segments):
        if i == 0:
            start_x = 0
            start_y = 0
            dx = seg_pts[0] - start_x
            dy = seg_hgts[0] - start_y
        else:
            start_x = seg_pts[i-1]
            start_y = seg_hgts[i-1]
            dx = seg_pts[i] - start_x
            dy = seg_hgts[i] - start_y

        m = 0 if dx == 0 else dy / dx

        for x in range(dx):
            y = int(m*x)
            crack_pts.append([start_x+x, start_y+y])

    # Attempt at rotation - TODO - Fix
    rotation = math.radians(rotation)
    for j in range(len(crack_pts)):
        x = crack_pts[j][0]
        y = crack_pts[j][1]
        crack_pts[j][0] = int(x * cos(rotation) + y * sin(rotation))
        crack_pts[j][1] = int(-x * sin(rotation) + y * cos(rotation))

    # Set the pixel's color
    for x, y in crack_pts:
        if len(mod_img_data[0][0]) == 4 and len(n_color) == 3:
            n_color = n_color + [255]

        for t in range(seg_thickness[i]):
            mod_img_data[center_y + y + t, center_x + x] = n_color
            mod_img_data[center_y + y - t, center_x + x] = n_color

    return mod_img_data


def add_incomplete_weld(center_x: int, center_y:int, cut_dir: str,
                        max_radius: int, defect_flag: int,
                        img_data, color, seed: int):
    '''

    :param center_x:
    :param center_y:
    :param max_width:
    :param defect_flag: boolean flag determining how far the weld is missing
    :return:
    '''

    mod_img_data = img_data.copy()
    cut_dir = cut_dir.lower()

    ##
    # Set reference points
    gap_radius = 2          # How wide is the gap b/w metal plates
    seem_radius = 4         # how wide is the seem the user is trying to weld
    buffer_from_weld = 6    # how far to set the 'edit start point' before the
                            #   weld

    ##
    # Define a range for values for a defect vs non-defect
    if defect_flag:
        max_into_weld = int(buffer_from_weld + WELD_TOL*2 + buffer_from_weld)
        min_into_weld = buffer_from_weld + WELD_TOL - (seem_radius-1)
        max_width_r = max_radius
        min_width_r = 2
    else:
        max_into_weld = int(buffer_from_weld + WELD_TOL - (seem_radius*1.5))
        min_into_weld = buffer_from_weld
        max_width_r = max_radius
        min_width_r = 0

    ###
    # Define the random traits associated with making the weld gap
    random.seed(seed)
    gap_depth = random.randint(min_into_weld, max_into_weld)
    gap_width_r = random.randint(min_width_r, max_width_r)

    ###
    # Create the square gap based on which direction the gap should be created.
    #   The direction guides where the pixel edits should start and what way
    #   they should go.
    #   Ex: if 'up', the gap starts below the specified center point going up.

    # Vertical weld
    if cut_dir == 'l':
        print("cut left")
        start_x = center_x + WELD_TOL + buffer_from_weld
        x_range = range(start_x - gap_depth, start_x)
        y_range = range(center_y-gap_width_r, center_y+gap_width_r)
    elif cut_dir == 'r':
        print("cut right")
        start_x = center_x - WELD_TOL - buffer_from_weld
        x_range = range(start_x, start_x + gap_depth)
        y_range = range(center_y-gap_width_r, center_y+gap_width_r)
    # Horizontal weld
    elif cut_dir == 'u':
        print("cut up")
        start_y = center_y - WELD_TOL - buffer_from_weld
        y_range = range(start_y, start_y + gap_depth)
        x_range = range(center_x - gap_width_r, center_x + gap_width_r)
    elif cut_dir == 'd':
        print("cut down")
        start_y = center_y + WELD_TOL + buffer_from_weld
        y_range = range(start_y - gap_depth, start_y)
        x_range = range(center_x - gap_width_r, center_x + gap_width_r)
    else:
        raise(f"ERROR: Invalid cut_direction ({cut_dir})")

    ###
    # Apply the pixel edits
    # TODO - consider adding 'linear gradient' method that takes a start
    #  color, end color, and dist from start to make a smooth gradient.
    for x in x_range:
        for y in y_range:

            seem_dim = y if cut_dir == 'u' or cut_dir == 'd' else x
            center_dim = center_y if cut_dir == 'u' or cut_dir == 'd' \
                                        else center_x

            # Set the pixel's color based on if it is the surface or weld seem
            if abs(center_dim - seem_dim) < seem_radius and \
                    color == norm_SURFACE:
                if center_dim-seem_radius < seem_dim < center_dim-gap_radius:
                    p_color = add_random_noise(LGRAY, 10)
                elif center_dim+gap_radius < seem_dim < center_dim+seem_radius:
                    p_color = add_random_noise(LGRAY, 10)
                else:
                    p_color = add_random_noise(DGRAY, 0)

            # Set pixel color outside of the weld to match the AO shadow
            # TODO - replace GRAY with the map name (ambient Occlusion)
            elif abs(center_dim - seem_dim) > WELD_TOL-2 and color == GRAY:
                if cut_dir == 'u' or cut_dir == 'd':
                    p_color = mod_img_data[y, x_range[0] - 1]
                else:
                    p_color = mod_img_data[y_range[0] - 1, x]
            else:
                p_color = color

            if len(mod_img_data[0][0]) == 4 and len(p_color) == 3:
                p_color = p_color + [255]

            try:
                mod_img_data[y, x] = p_color
            except:
                print("Help")

    return mod_img_data


def define_weld_pts(pixel_spacing: int):
    '''
    Define the x,y coordinates for all weld locations under review. Points
    are relative to the (0,0) point in the Top-Left part of a bitmap file.


    :param pixel_spacing:
    :return: list of [x,y] entries specifying offsets to take screenshots of.
    '''

    # Locations for horizontal welds
    y_axis_intersects = LINES_HORZ[1:2]
    # [1365, 2732]

    x_locs = range(0, 4096, pixel_spacing)

    pts_01 = []
    for weld_row_y in y_axis_intersects:
        for x in x_locs:
            pts_01.append([x, weld_row_y])

    # Locations for vertical welds
    x_axis_intersects = LINES_VERT      # Intersection of x-axis
    # [[308, 1333, 2358, 3382],
    #  [141, 1169, 2193, 3218],
    #  [651, 1675, 2699, 3723]]

    pts_02 = []
    for weld_col_x in x_axis_intersects[0]:
        y_locs = range(0, y_axis_intersects[0], pixel_spacing)
        for y in y_locs:
            pts_02.append([weld_col_x, y])

    for weld_col_x in x_axis_intersects[1]:
        y_locs = range(y_axis_intersects[0], y_axis_intersects[1],
                       pixel_spacing)
        for y in y_locs:
            pts_02.append([weld_col_x, y])

    for weld_col_x in x_axis_intersects[2]:
        y_locs = range(y_axis_intersects[1], 4096, pixel_spacing)
        for y in y_locs:
            pts_02.append([weld_col_x, y])

    return pts_01 + pts_02