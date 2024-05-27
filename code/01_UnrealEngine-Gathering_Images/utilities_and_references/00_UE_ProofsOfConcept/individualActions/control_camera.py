'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import time
import unreal

# ar_lib = unreal.ARLibrary()
#
# cam = unreal.getARTexture()
# ar_lib.get_camera_image()
#
# cam = unreal.CameraActor(name='CineCameraActor')


# Get main camera   - Confirmed if CAMERA is selected
actor = unreal.EditorLevelLibrary.get_selected_level_actors()
print(actor)

# Learn about camera from it's 'scene_component' class
scene = actor[0].scene_component
#home = scene.get_world_location()
home = unreal.Vector(x=-150, y=-60, z=150)
print(f"HOME Camera location: {home}")

# Move the camera either one of two ways:
#   A) Set the actual location & rotation   - Confirmed
#       scene.set_world_location(new_location=, sweep=False, teleport=True)
#       scene.set_world_rotation(new_rotation=, sweep=False, teleport=True)
#
#   B) Set relative offset
#       scene.add_world_offset()
#       scene.add_world_rotation()

'''
Below works, but from a single script call, the effects will only be applied 
IF there is an offset by the end of the script. If you move, then move back, 
nothing in the editor will change.
'''

# Move first
x = home.x
y = home.y
z = home.z
scene.set_world_location(new_location=[x+30,y,z], sweep=False, teleport=True)
current_loc = scene.get_world_location()
print(f"ALTERED Camera location: {current_loc}")

#time.sleep(2)


# Save changes
# editor = unreal.EditorLoadingAndSavingUtils
# editor.save_dirty_packages(True, True)

# unreal.EditorUtilityLibrary.set_actor_selection_state(scene, True)
# unreal.EditorLevelLibrary.set_level_dirty()

# unreal.EditorAssetLibrary.save_loaded_asset(actor)
# unreal.EditorLevelLibrary.save_current_level()

# Move Second
time.sleep(2)
x = home.x
y = home.y
z = home.z
scene.set_world_location(new_location=[x-30,y,z], sweep=False, teleport=True)
current_loc = scene.get_world_location()
print(f"ALTERED Camera location: {current_loc}")

# scene.set_world_location(new_location=home, sweep=False, teleport=True)

