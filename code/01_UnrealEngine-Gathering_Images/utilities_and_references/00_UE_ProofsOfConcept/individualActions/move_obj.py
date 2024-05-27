'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import random
import unreal

# TODO - Implement a cycle to return all actors HOME before moving.

#####
home_path = '/Game/_MENG_Project/04_ImgPipeline0/'

# Code to print out all or specific actors in a loaded level    - Confirmed
editor = unreal.EditorLevelLibrary()
actors = editor.get_all_level_actors()

objects = []
CENTER_POINT = None
MODEL_POINT = None
LOC_POINTS = []

for y in actors:
    #print(f"Actor: {y.get_name()}")

    if 'cat' in y.get_name() or 'Cat' in y.get_name():
        objects.append(y)
        print(f"*** DEBUG: Found cat - {y.get_name()}")

    if 'CENTER_POINT' == y.get_name():
        CENTER_POINT = y
        print("*** DEBUG: Found CENTER_POINT")

    if 'MODEL_POINT' == y.get_name():
        CENTER_POINT = y
        print("*** DEBUG: Found MODEL_POINT")

    if 'LOC' in y.get_name():
        LOC_POINTS.append(y)
        print(f"*** DEBUG: Found actor location {y.get_name()}")


def unreal_vector_to_list(vec: unreal.Vector):
    #print(vec)
    #print(f"{vec.x}/{vec.y}/{vec.z}")
    return [vec.x, vec.y, vec.z]


# #############
# # Execute Simple single move
# for x in range(len(objects)):
#
#     model = objects[x]
#     cur_loc = unreal_vector_to_list(
#         model.static_mesh_component.relative_location)
#     print(f"*** DEBUG: Chose {model.get_name()} at {cur_loc}")
#
#     dest = random.randint(0, len(LOC_POINTS))
#     if dest < len(LOC_POINTS):
#         dest = LOC_POINTS[dest]
#     elif dest == len(LOC_POINTS):
#         dest = CENTER_POINT
#     elif dest > len(LOC_POINTS):
#         dest = MODEL_POINT
#
#     new_loc = unreal_vector_to_list(
#         dest.static_mesh_component.relative_location)
#     print(f"*** DEBUG: Move to {dest.get_name()} at {new_loc}")

#############
# Execute move - cycle objects around different points
chosen_models = random.sample(range(len(objects)), k=3)
chosen_locs = random.sample(range(len(LOC_POINTS)+1), k=3)

print(f"*** DEBUG: Can choose models {chosen_models} and locs {chosen_locs}")

for x in chosen_models:

    model = objects[x]
    dest = random.sample(chosen_locs, k=1)[0]
    chosen_locs.remove(dest)

    if dest < len(LOC_POINTS):
        dest = LOC_POINTS[dest]
    elif dest == len(LOC_POINTS):
        dest = CENTER_POINT
    elif dest > len(LOC_POINTS):
        dest = MODEL_POINT


    cur_loc = unreal_vector_to_list(
                    model.static_mesh_component.relative_location)
    new_loc = unreal_vector_to_list(
                    dest.static_mesh_component.relative_location)
    print(f"*** DEBUG: Chose {model.get_name()} at {cur_loc}. "
          f"Move to {dest.get_name()} at {new_loc}")

    dest_vect = unreal.Vector(x=dest.static_mesh_component.relative_location.x,
                              y=dest.static_mesh_component.relative_location.y,
                              z=dest.static_mesh_component.relative_location.z\
                                  + 70)
    model.static_mesh_component.set_editor_property("relative_location",
                        dest_vect)



