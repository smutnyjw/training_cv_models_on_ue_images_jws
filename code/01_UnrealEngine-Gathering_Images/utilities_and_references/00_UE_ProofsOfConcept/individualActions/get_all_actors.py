'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import unreal
import random


#####
# Code to print out all or specific actors in a loaded level    - Confirmed
editor = unreal.EditorLevelLibrary()
actors = editor.get_all_level_actors()
#print(actors)


for y in actors:
    print(f"Actor: {y.get_name()}")
    if y.get_name() == "SunSky_2":
        skysun = y

    if y.get_name() == "PointLight_1":
        p_light1 = y

    if y.get_name() == "PointLight_2":
        p_light2 = y

    if y.get_name() == "SpotLight_1":
        s_light = y


###############################################
#   Useful changes to Actors to this project
###############################################


####
# Toggle the main spotlight
#   Kinda works. The editor lighting needs to be rebuilt if done.
# https://docs.unrealengine.com/4.27/en-US/PythonAPI/class/SpotLightComponent.html?highlight=spotlightcomponent#unreal.SpotLightComponent

if s_light:
    comp = s_light.get_component_by_class(unreal.SpotLightComponent)
    #comp.set_editor_property("hidden_in_game", False)
    comp.set_editor_property("Intensity", 0.0)


####
# Randomly change the color and Intensity of point lights   - CONFIRMED
# https://docs.unrealengine.com/4.27/en-US/PythonAPI/class/PointLightComponent.html

if p_light1 and p_light2:
    comp1 = p_light1.get_component_by_class(unreal.PointLightComponent)
    c1 = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
    i1 = random.randint(0, 40)
    comp1.set_editor_property("Intensity", i1)
    comp1.set_editor_property("light_color", c1)
    print("*** Changed PointLight_1")

    comp2 = p_light2.get_component_by_class(unreal.PointLightComponent)
    c2 = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
    i2 = random.randint(0, 40)
    comp2.set_editor_property("Intensity", i2)
    comp2.set_editor_property("light_color", c2)
    print("*** Changed PointLight_2")

####
# Turn OFF SkySun lighting is applied to the editor - CONFIRMED
#   True = SkySun is Hidden
#   False = SkySun is active

if not skysun.is_temporarily_hidden_in_editor():
    print(f"SkySun Status1 = {skysun.is_temporarily_hidden_in_editor()}")
    # y.set_actor_hidden(True)
    # y.set_actor_hidden_in_game(True)
    # y.set_editor_property("hidden", True)
    # y.is_temporarily_hidden_in_editor(True)
    skysun.set_is_temporarily_hidden_in_editor(True)    # WORKS
    print(f"SkySun Status2 = {skysun.is_temporarily_hidden_in_editor()}")


###############################################
#   Misc attempts to edit Actors
###############################################

spot_light = []
point_light = []
for x in actors:
    #print(f"Obj = {x}")
    if x.get_component_by_class(unreal.CameraComponent) != None:
        cam = x
        #print(f"CAMERA = {x.get_component_by_class(unreal.CameraComponent)}")

    # https://docs.unrealengine.com/4.27/en-US/PythonAPI/class/SkyLightComponent.html#unreal.SkyLightComponent
    if x.get_component_by_class(unreal.SkyLightComponent):
        sun = x
        # print(f"SUN LIGHT ="
        #         f"{x.get_component_by_class(unreal.SkyLightComponent)}")

    #
    if x.get_component_by_class(unreal.SpotLightComponent):
        spot_light.append(x)
        # print(f"Got SPOT LIGHT {len(spot_light)}")

    # https://docs.unrealengine.com/4.27/en-US/PythonAPI/class/PointLight.html?highlight=pointlight#unreal.PointLight
    if x.get_component_by_class(unreal.PointLightComponent) and  \
        not x.get_component_by_class(unreal.SpotLightComponent):
        point_light.append(x)
        # print(f"Got POINT LIGHT {len(point_light)}")

#####
# Toggle visibility of the SkyLight COMPONENT (not actor)...
#   If enabled and ON will overcast a shadow
if False:
    sun = sun.get_component_by_class(unreal.SkyLightComponent)

    vis = 0 if sun.get_editor_property("visible") else 1
    sun.set_editor_property("visible", vis)    # Does not do as intended
    sun.recapture_sky()

#####
# Failed attempt to take screenshot specifically from the CineCameraComponent
# auto_lib = unreal.AutomationLibrary()
#
# # https://docs.unrealengine.com/4.27/en-US/PythonAPI/class/AutomationScreenshotOptions.html#unreal.AutomationScreenshotOptions
# opt = unreal.AutomationScreenshotOptions(
#     resolution=[256, 256],
#     delay=0.2,
#     override_override_time_to=False,
#     override_time_to=0.0,
#     disable_noisy_rendering_features=True,
#     disable_tonemapping=True,
#     view_settings=None,
#     visualize_buffer='None',
#     #tolerance=,
#     #tolerance_amount=[],
#     maximum_local_error=0.1,
#     maximum_global_error=0.02,
#     ignore_anti_aliasing=True,
#     ignore_colors=False
# )
#
# la = unreal.LatentActionInfo()
#
# auto_lib.take_automation_screenshot_at_camera(
#     editor,
#     la,
#     cam,
#     "main camera",
#     "screenshot from main camera",
#     opt
# )





