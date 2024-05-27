'''
File:   
Author: 
Date:   
Description: 
    A narrow proof-of-concept script that will rotate a UE4 CineCamera actor
    around an object then take screenshots. Rotation will occur in orbit
    around a particular environment object. Changes will be in distance,
    azimuth, and aspect dimensions

    Rotation control will later be adapted in greater script.

Other Notes:

In UE editor:
1. Set camera to be child of poi_obj
2. Set camera lookatTrackingSettings.EnableLookatTracking to
ActorToTrack=POI_obj
    unreal.CameraLookatTrackingSettings()
3. Set camera to the first location to apply rotations too.
4. Set the Editor perspective to your desired rotating camera.
5. Double check that the POI_OBJ scales are 1. This will throw off Camera
movements.


'''

import unreal
from math import acos, pi, cos, sin

# class MoveCamera(object):
#     def __init__(self, iterations, obj: dict, cam_home):
#         # Threading setup
#         self.frame_count = 0
#         self.max_frame_count = 50
#         self.action_count = 0
#         self.num_actions = iterations
#
#         self.slate_post_tick_handle = None
#
#         self.editor = obj["level_editor"]
#
#         # Camera setup
#         self.camera_obj = obj["camera"]
#         self.cam_scene = self.camera_obj.scene_component
#         # cam_loc = self.camera_obj.scene_component.get_world_location()
#         self.cam_home = unreal.Vector(x=cam_home[0],
#                                       y=cam_home[1],
#                                       z=cam_home[2])
#
#         # Point of Interest for camera to focus on
#         self.poi_obj = obj["point_of_interest_obj"]
#         self.poi_obj_loc = self.poi_obj.static_mesh_component.relative_location
#         self.poi_angles = self.calc_cam_rotation_track()
#
#         self.set_environment()
#
#     def set_environment(self):
#         print("TODO - ::set_environment()")
#
#         # Set initial camera
#         self.teleport_camera_home(home=self.cam_home)
#         print(f"HOME relative Camera location: {self.cam_home}")
#
#
#     def start(self):
#         self.editor.clear_actor_selection_set()
#         self.editor.set_selected_level_actors([self.camera_obj])
#         if self.frame_count != 0:
#             print("Please wait until first call is done.")
#         else:
#             self.slate_post_tick_handle = \
#                 unreal.register_slate_post_tick_callback(self.tick)
#             self.frame_count = 0
#
#             print("Test")
#
#     def tick(self, delta_time: float):
#         # Potentially end the thread
#         if self.frame_count > self.max_frame_count:
#             unreal.log_error("<<<<< Hit max_frame_count. Terminating <<<<<<")
#             # End tick and any changes
#             unreal.unregister_slate_post_tick_callback(
#                 self.slate_post_tick_handle)
#
#
#         #######################################################
#         #           Perform Actions at ticks
#         #######################################################
#
#         # 1) Reset at end of actions or as a protective measure
#         if self.action_count == self.num_actions\
#                 or self.frame_count > self.max_frame_count - 1:
#             # Reset before the end of the thread
#             self.teleport_camera_home(home=self.cam_home)
#             self.action_count += 1
#
#         # 2) Perform action sequence
#         elif self.action_count < self.num_actions:
#             TICK_BUFFER = 5
#             if self.frame_count % TICK_BUFFER == 0:
#                 self.run_camera_rotation_seq(self.action_count)
#
#                 self.action_count += 1
#
#
#         self.frame_count += 1
#
#     #######################################################
#     #           Test Sequences
#     #######################################################
#
#     def run_camera_rotation_seq(self, action_num):
#         (f"---- Performing Action {action_num}")
#
#         # if action_num == 0:
#         #     print("*** DEBUG: Skip first action")
#         #     return
#
#         # Change the Editor
#
#
#         try:
#             unreal.log(f"*** Rotation seq #{action_num}")
#             cam_loc = self.poi_angles[action_num]
#             self.teleport_camera(_x=cam_loc[0],
#                                  _y=cam_loc[1],
#                                  _z=cam_loc[2])
#
#             # Take a screenshot
#             self.take_HighResShot2()
#
#         except IndexError:
#             unreal.log_warning(f"*** WARN: Attempting too many camera "
#                               f"rotations. "
#                   f"Only {len(self.poi_angles)} camera locations are "
#                   f"specified.")
#
#
#
#     #######################################################
#     #           CineCameraActor Methods
#     #######################################################
#
#     def take_HighResShot2(self):
#         unreal.log("***** Take screenshot of current Editor")
#         unreal.SystemLibrary.execute_console_command(
#             self.editor.get_editor_world(), "HighResShot 2")
#
#     def teleport_camera(self, _x, _y, _z):
#         # Move first
#         self.cam_scene.set_relative_location(new_location=[_x, _y, _z],
#                                           sweep=False,
#                                           teleport=True)
#         self.cam_scene.set_relative_rotation(new_rotation=[0, 0, 0],
#                                              sweep=False,
#                                              teleport=True)
#         current_loc = self.cam_scene.relative_location
#         unreal.log_warning(f"ALTERED releative Camera location: {current_loc}")
#
#     def teleport_transform_camera(self, d_x, d_y, d_z):
#         move = unreal.Vector(x=d_x, y=d_y, z=d_z)
#         trans = unreal.Transform(location=move)
#
#         # trans = unreal.Transform()
#         # trans.transform_location(move)
#
#         self.cam_scene.add_local_transform(delta_transform=trans,
#                                            sweep=False,
#                                            teleport=True)
#         self.cam_scene.set_world_scale3d(unreal.Vector(x=1.0, y=1.0, z=1.0))
#         current_loc = self.cam_scene.relative_location
#         unreal.log_warning(f"ALTERED relative Camera location: {current_loc}")
#
#         # TODO - need to find out if this is worth trying. Will need to 'move
#         #  cam' off of the delta in curren_loc and expected_loc
#
#     def teleport_camera_home(self, home):
#         unreal.log_warning("*** Return camera to HOME position ***")
#         self.teleport_camera(_x=home.x,
#                              _y=home.y,
#                              _z=home.z)
#
#     def calc_cam_rotation_track(self):
#         end_cam_locs = []
#         diff_cam_obj_dist = []
#
#         #set your own target here
#         target = self.poi_obj
#         cam = self.camera_obj
#         t_loc = self.poi_obj.static_mesh_component.relative_location
#         t_loc_x = t_loc.x
#         t_loc_y = t_loc.y
#         cam_loc = self.camera_obj.scene_component.relative_location
#         cam_loc_x = cam_loc.x
#         cam_loc_y = cam_loc.y
#
#
#         R = pow(pow(t_loc_x-cam_loc_x,2) + pow(t_loc_y-cam_loc_y,2), 0.5) # Radius
#
#         # The different radii range
#         radius_range = [R, R+100]
#
#
#         init_angle  = (1-2*bool((cam_loc_y-t_loc_y)<0))*acos((cam_loc_x-t_loc_x)/R)-2*pi*bool((cam_loc_y-t_loc_y)<0) # 8.13 degrees
#         target_angle = (pi/2 - init_angle) # Go 90-8 deg more
#         num_steps = 6 #how many rotation steps
#
#         for r in radius_range:
#             for x in range(num_steps):
#                 alpha = init_angle + (x)*target_angle/num_steps
#                 #cam.rotation_euler[2] = pi/2 + alpha #
#                 #cam.location.x = t_loc_x+cos(alpha)*r
#                 #cam.location.y = t_loc_y+sin(alpha)*r
#
#                 cam_x = round(t_loc_x + cos(alpha) * r, 2)
#                 cam_y = round(t_loc_y + sin(alpha) * r, 2)
#
#                 #self.teleport_camera(_x=cam_x, _y=cam_y, _z=)
#
#                 # Log locations
#                 end_cam_locs.append([cam_x, cam_y, r])
#
#
#                 diff_x = round(cam_x-t_loc_x, 1)
#                 diff_y = round(cam_y-t_loc_y, 1)
#                 diff_cam_obj_dist.append([diff_x, diff_y])
#
#         unreal.log(f"*** DEBUG: End rotation locations ({len(end_cam_locs)})"
#               f"\n{end_cam_locs}\n\tCam v Obj distances: {diff_cam_obj_dist}")
#         return end_cam_locs

def teleport_camera(cam, _x, _y, _z):
    # Move first
    cam.scene_component.set_relative_location(new_location=[_x, _y, _z],
                                      sweep=False,
                                      teleport=True)
    cam.scene_component.set_relative_rotation(new_rotation=[0, 0, 0],
                                         sweep=False,
                                         teleport=True)
    current_loc = cam.scene_component.relative_location
    unreal.log_warning(f"ALTERED releative Camera location: {current_loc}")

def rotate_camera(cam, roll, pitch, yaw):

    pitch = pitch - 90
    yaw = yaw - 180

    rot = unreal.Rotator(roll=roll, pitch=pitch, yaw=yaw)
    cam.scene_component.set_relative_rotation(new_rotation=rot,
                                              sweep=False,
                                              teleport=True)
    current_rot = cam.scene_component.relative_rotation
    unreal.log_warning(f"ALTERED releative Camera rotation: {current_rot}")

#####
# Code to print out all or specific actors in a loaded level    - Confirmed
editor = unreal.EditorLevelLibrary()
actors = editor.get_all_level_actors()
#print(actors)


for y in actors:
    #print(f"Actor: {y.get_name()}")

    if y.get_name() == "Main_Camera":
        # Ensure that the camera Actor is selected for screenshots
        cam = y
        unreal.log("*** DEBUG: Found Camera actor")

    if y.get_name() == "12221_Cat_v1_l4_17":
        poi_obj = y
        unreal.log("*** DEBUG: Found 12221_Cat_v1_l4 actor")

actor_dic = {"level_editor": editor,
                "camera": cam,
                "point_of_interest_obj": poi_obj}


cam_home = [0, -300, 200]

track_set = unreal.CameraLookatTrackingSettings(enable_look_at_tracking=True,
                                                actor_to_track=poi_obj)
# cam.set_editor_property.lookat_tracking_settings(track_set)
print(cam)
print(cam.set_editor_property("lookat_tracking_settings", track_set))



# NOTE:
#   Cannot use 'linear_transform' since 'cam_track' causes incorrect movement.
# test1.teleport_transform_camera(d_x=-30, d_y=0, d_z=0)
#test1.teleport_transform_camera(d_x=0, d_y=90, d_z=0)

def spherical_to_cartisen_deg(rho: int=0, alpha: float=0, phi: float=0):
    # Convert inputs to radians
    alpha = alpha * pi/180
    phi = phi * pi/180

    # Convert Spherical coordinates to cartesian
    x = rho*sin(phi)*cos(alpha) #212
    y = rho*sin(phi)*sin(alpha) #212
    z = rho*cos(phi)        #0

    x = round(x, 2)
    y = round(y, 2)
    z = round(z, 2)

    return x, y, z

start_loc_poi = [poi_obj.static_mesh_component.relative_location.x,
                   poi_obj.static_mesh_component.relative_location.y,
                   poi_obj.static_mesh_component.relative_location.z]
start_loc_cam = [cam.scene_component.relative_location.x,
                   cam.scene_component.relative_location.y,
                   cam.scene_component.relative_location.z]

# Below are acceptable range of values given the present setup of the envi
range_rho = [200]
range_alpha = [60, 200]
range_phi = [0, 90]

rho = 200
alpha = 180 #[-180, 0]
phi = 80    #[0, 90]

x, y, z = spherical_to_cartisen_deg(rho=rho, alpha=alpha, phi=phi)

teleport_camera(cam,
                _x=x,#+start_loc[0],
                _y=y,#+start_loc[1],
                _z=z,#+start_loc[2]
                )
rotate_camera(cam,
                roll=0.0,
                pitch=phi,
                yaw=alpha)

end_loc = cam.scene_component.relative_location

print(f"START: {start_loc_cam[0]}/{start_loc_cam[1]}/{start_loc_cam[2]}\n"
      f"xyz: {x}/{y}/{z}\n"
      f"VERIFY: {end_loc.x}/{end_loc.y}/{end_loc.z}")

