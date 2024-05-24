'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import os
import random
from copy import deepcopy

import cv2
import lib_cmn as cmn

base_path = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project"
input_img_path = os.path.join(base_path, "materials", "parent_assets")
output_path = os.path.join(base_path, "10_ImagePipeline_weld", "map_sets")
output_map_path = {
                "non_defect": os.path.join(output_path, "set02-nonDefect-mod"),
                "porosity": os.path.join(output_path, "set03-porosity"),
                "crack": os.path.join(output_path, "set04-crack"),
                "incomplete": os.path.join(output_path, "set05-incomplete")
        }

parent_maps = {
                "normal": "T_Welded_metal_normal.TGA.bmp",
                "ao": "T_Welded_metal_ambientocclusion.bmp",
                "metallic": "T_Welded_metal_metallic.bmp",
                "roughness": "T_Welded_metal_roughness.bmp"
        }

mod_locs = cmn.define_weld_pts(pixel_spacing=400,
                               dist_from_border=125)


# ======================================================================


def output_porosity_map_set(mod_locs: list,
                            parent_maps: dict,
                            output_path: str,
                            defect_flag: int):
    print("--- Run output_porosity_map_set() ---")

    seeds = random.sample(range(1, 9999), len(mod_locs))

    for key in parent_maps:
        ue_map = parent_maps[key]
        img = cv2.imread(os.path.join(input_img_path, ue_map),
                         cv2.IMREAD_UNCHANGED)

        i = 0
        for x, y, weld_dir in mod_locs:
            # Add a modification at each location on a map
            try:
                random.seed(seeds[i])
                img = cmn.add_oval(center_x=x+random.randint(-3, 3),
                                   center_y=y+random.randint(-3, 3),
                                   radius_x=random.randint(2, 6),
                                   radius_y=random.randint(2, 6),
                                   defect_flag=defect_flag,
                                   img_data=img,
                                   color=cmn.defect_color_dict["porosity"][key],
                                   seed=seeds[i])
                i += 1
            except:
                print(f"*** WARN-porosity: skipping porosity location "
                      f"[{x},{y}]")
                continue

        ###
        # Output the new map
        cv2.imwrite(os.path.join(output_path, ue_map + ".porosity.bmp"),
                    img)

    random.seed(None)


def determine_crack_rotation(weld_dir: str):
    if 'h' in weld_dir.lower():
        rotation = random.sample([0, 180], 1)[0]
    elif 'v' in weld_dir.lower():
        rotation = 270
    else:
        raise (f"*** ERROR - determine_crack_rotation: Invalid 'weld_dir' "
               f"value {weld_dir}. Must be [horizontal, vertical]")

    return rotation


def output_crack_map_set(mod_locs: list,
                            parent_maps: dict,
                            output_path: str,
                            defect_flag: int):
    print("--- Run output_crack_map_set() ---")

    seeds = random.sample(range(1, 9999), len(mod_locs))

    for key in parent_maps:
        ue_map = parent_maps[key]
        img = cv2.imread(os.path.join(input_img_path, ue_map),
                         cv2.IMREAD_UNCHANGED)

        i = 0
        for x, y, weld_dir in mod_locs:
            random.seed(seeds[i])
            rotation = determine_crack_rotation(weld_dir)

            # Add a modification at each location on a map
            try:
                img = cmn.add_crack(center_x=x,
                                    center_y=y,
                                    length=random.randint(30, 80),
                                    segments=random.randint(3, 10),
                                    tol=random.randint(4, int(cmn.WELD_TOL)-3),
                                    thickness=random.randint(2, 4),
                                    rotation=rotation,
                                    defect_flag=defect_flag,
                                    color=cmn.defect_color_dict["crack"][key],
                                    img_data=img,
                                    seed=seeds[i])
                i += 1
            except:
                print(f"*** WARN-crack: skipping crack location [{x},{y}]")
                continue

        ###
        # Output the new map
        cv2.imwrite(os.path.join(output_path, ue_map + ".crack.bmp"),
                    img)

    random.seed(None)


def determine_incomplete_dir(weld_dir: str):
    if 'h' in weld_dir.lower():
        cut_dir = random.sample(['u', 'd'], 1)[0]
    elif 'v' in weld_dir.lower():
        cut_dir = random.sample(['l', 'r'], 1)[0]
    else:
        raise (f"*** ERROR: Invalid 'weld_dir' value "
               f"{weld_dir}. Must be [horizontal, vertical]")

    return cut_dir


def output_incomplete_map_set(mod_locs: list,
                                parent_maps: dict,
                                output_path: str,
                                defect_flag: int):
    print("--- Run output_incomplete_map_set() ---")

    seeds = random.sample(range(1, 9999), len(mod_locs))

    for key in parent_maps:
        ue_map = parent_maps[key]
        img = cv2.imread(os.path.join(input_img_path, ue_map),
                         cv2.IMREAD_UNCHANGED)
        i = 0
        for x, y, weld_dir in mod_locs:
            random.seed(seeds[i])
            cut_dir = determine_incomplete_dir(weld_dir)

            # Add a modification at each location on a map
            try:
                img = cmn.add_incomplete_weld(
                                center_x=x,
                                center_y=y,
                                cut_dir=cut_dir,
                                max_radius=random.randint(5, 30),
                                defect_flag=defect_flag,
                                img_data=img,
                                color=cmn.defect_color_dict["incomplete"][key],
                                seed=seeds[i]
                              )
                i += 1
            except:
                print(f"*** WARN-incomplete: skipping incomplete location "
                      f"[{x},{y}]")
                continue

        ###
        # Output the new map
        cv2.imwrite(os.path.join(output_path, ue_map + ".incomplete.bmp"),
                    img)

    random.seed(None)


def output_non_defect_map_set(mod_locs: list,
                                parent_maps: dict,
                                output_path: str):
    print("--- Run output_non_defect_map_set() ---")
    mod_locs = deepcopy(mod_locs)
    seeds = random.sample(range(1, 9999), len(mod_locs))

    # Decide which defect to apply to which loc
    for i in range(len(mod_locs)):
        r = random.randint(0, len(cmn.defect_color_dict.keys())-1)
        key = list(cmn.defect_color_dict.keys())[r]

        mod_locs[i].append(key)

    # Edit each map with the randomly selected defect
    for key in parent_maps:
        ue_map = parent_maps[key]
        img = cv2.imread(os.path.join(input_img_path, ue_map),
                         cv2.IMREAD_UNCHANGED)

        i = 0
        for x, y, weld_dir, mod in mod_locs:
            # Add a modification to this location on a map
            random.seed(seeds[i])
            if mod == "porosity":
                try:
                    img = cmn.add_oval(center_x=x+random.randint(-3, 3),
                                   center_y=y+random.randint(-3, 3),
                                   radius_x=random.randint(2, 4),
                                   radius_y=random.randint(2, 4),
                                   defect_flag=False,
                                   img_data=img,
                                   color=cmn.defect_color_dict[mod][key],
                                   seed=seeds[i])
                except:
                    print(f"*** WARN-mixed: skipping porosity location "
                          f"[{x},{y}]")
                    continue

            elif mod == "crack":
                try:
                    rotation = determine_crack_rotation(weld_dir)

                    img = cmn.add_crack(center_x=x+random.randint(-2, 2),
                                        center_y=y+random.randint(-2, 2),
                                        length=random.randint(30, 80),
                                        segments=random.randint(3, 10),
                                        tol=random.randint(4,
                                                           int(cmn.WELD_TOL)-3),
                                        thickness=random.randint(2, 4),
                                        rotation=rotation,
                                        defect_flag=False,
                                        color=cmn.defect_color_dict[mod][key],
                                        img_data=img,
                                        seed=seeds[i])
                except:
                    print(f"*** WARN-mixed: skipping crack location [{x},{y}]")
                    continue

            elif mod == "incomplete":
                try:
                    cut_dir = determine_incomplete_dir(weld_dir)

                    img = cmn.add_incomplete_weld(
                                        center_x=x,
                                        center_y=y,
                                        cut_dir=cut_dir,
                                        max_radius=random.randint(3, 30),
                                        defect_flag=False,
                                        img_data=img,
                                        color=cmn.defect_color_dict[mod][key],
                                        seed=seeds[i]
                                      )
                except:
                    print(f"*** WARN-mixed: skipping incomplete location "
                          f"[{x},{y}]")
                    continue


            else:
                raise ("*** ERROR - output_non_defect_map_set(): "
                       f"Could not match modification {mod}")

            i += 1

        ###
        # Output the new map
        cv2.imwrite(os.path.join(output_path, ue_map + ".mixed.bmp"),
                    img)

    random.seed(None)



# ======================================================================


output_non_defect_map_set(mod_locs=mod_locs,
                            parent_maps=parent_maps,
                            output_path=output_map_path["non_defect"])

output_porosity_map_set(mod_locs=mod_locs,
                        parent_maps=parent_maps,
                        output_path=output_map_path["porosity"],
                        defect_flag=True)

output_crack_map_set(   mod_locs=mod_locs,
                        parent_maps=parent_maps,
                        output_path=output_map_path["crack"],
                        defect_flag=True)

output_incomplete_map_set(  mod_locs=mod_locs,
                            parent_maps=parent_maps,
                            output_path=output_map_path["incomplete"],
                            defect_flag=True)





