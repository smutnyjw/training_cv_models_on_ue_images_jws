'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''


import unreal

#####
# Code to print out all or specific actors in a loaded level    - Confirmed
editor = unreal.EditorLevelLibrary()
actors = editor.get_all_level_actors()

for y in actors:
    if 'SunSky' in y.get_name():
        sunsky = y
        print("*** DEBUG: Found Sunsky object")

print("-------------------")
print(sunsky.get_components_by_class(unreal.SceneComponent))
print(sunsky.get_component_by_class(unreal.SkyLightComponent))

skylight = sunsky.get_component_by_class(unreal.SkyLightComponent)

print("-------------------")
print(sunsky)
print(sunsky.get_editor_property("Latitude"))
print(sunsky.get_editor_property("Longitude"))
print(sunsky.get_editor_property("NorthOffset"))
print(sunsky.get_editor_property("TimeZone"))
print(sunsky.get_editor_property("Month"))
print(sunsky.get_editor_property("Day"))
print(sunsky.get_editor_property("SolarTime"))


import random
i = random.randint(12, 23)
sunsky.set_editor_property("SolarTime", i)