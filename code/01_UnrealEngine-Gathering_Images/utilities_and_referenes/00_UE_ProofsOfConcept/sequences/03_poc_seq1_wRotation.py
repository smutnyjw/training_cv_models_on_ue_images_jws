'''
File:   
Author: 
Date:   
Description:
    Base script that will take x HighResShot2 screenshots of the Unreal
    Editor at different camera locations then return the camera to the
    starting HOME point.

    Actions are based on the '.register_slate_post_tick_callback()' method
    to create a new asynchronous process, where the '.tick()' method mimics a
    separate thread (NOTE: UE4 scripting does not support multi-threading,
    this meerly ACTS like a multi-thread.

Other Notes:
    Do not use time.sleep(). It will throw everything off


In UE editor:
1. Set camera to be child of poi_obj
2. Set the Editor perspective to your desired rotating camera.
3. Double check that the POI_OBJ scales are 1. This will throw off Camera
movements.

'''

import unreal
import random
from math import acos, pi, cos, sin


def spherical_to_cartisen_deg(rho: int = 0,
                              alpha: float = 0,
                              phi: float = 0):
    # Convert inputs to radians
    alpha = alpha * pi/180
    phi = phi * pi/180

    # Convert Spherical coordinates to cartesian
    x = rho * sin(phi) * cos(alpha)
    y = rho * sin(phi) * sin(alpha)
    z = rho * cos(phi)

    x = round(x, 2)
    y = round(y, 2)
    z = round(z, 2)

    return x, y, z


class MoveCamera(object):
    def __init__(self, iterations, material_paths: list, obj: dict, cam_home):
        # Threading setup
        self.frame_count = 0
        self.max_frame_count = 300
        self.action_count = 0
        self.num_actions = iterations

        self.slate_post_tick_handle = None

        self.editor = obj["level_editor"]

        # Camera setup
        self.camera_obj = obj["camera"]
        self.cam_scene = self.camera_obj.scene_component
        self.cam_home = unreal.Vector(x=cam_home[0],
                                      y=cam_home[1],
                                      z=cam_home[2])

        # Point of Interest for camera to focus on
        self.poi_obj = obj["point_of_interest_obj"]
        self.poi_obj_loc = self.poi_obj.static_mesh_component.relative_location
        self.ss_cam_locs = []
        self.ss_cam_angles = []
        self.ss_cur_cam_position = 0

        # Material_0 setup  - BLUE_block
        self.mat0 = self.setup_material(material_paths[0])
        self.mat0_params = self.get_material_params(self.mat0)

        # Material_1 setup  - STAR_block
        self.mat1 = self.setup_material(material_paths[1])
        self.mat1_params = self.get_material_params(self.mat1)

        # Material_2 setup  - SQ_Rust_block
        self.mat2 = self.setup_material(material_paths[2])
        self.mat2_params = self.get_material_params(self.mat2)

        # Point Lights
        self.point_light1 = obj["point_light1"]
        self.point_light2 = obj["point_light2"]

        self.set_environment()

    def set_environment(self):
        unreal.log("TODO - ::set_environment()")

        # Set initial camera location
        self.teleport_camera_home(home=self.cam_home)
        print(f"HOME Camera location: {self.cam_home}")

        # Set camera to focus on the target
        track_set = unreal.CameraLookatTrackingSettings(
                                enable_look_at_tracking=True,
                                actor_to_track=self.poi_obj)
        cam.set_editor_property("lookat_tracking_settings", track_set)

        # Turn off SunSkylight
        unreal.log("TODO - ::set_environment() - Turn off SunSkyLight")

        # Set initial light intensities & colors
        unreal.log("TODO - ::set_environment() - Set Initial Light intensities")

    def start(self, seq: int):
        self.editor.clear_actor_selection_set()
        self.editor.set_selected_level_actors([self.camera_obj])
        if self.frame_count != 0:
            print("Please wait until first call is done.")
        else:
            if seq == 1:
                unreal.log("*** DEBUG: Registering ::tick_01 callback")
                self.slate_post_tick_handle = \
                    unreal.register_slate_post_tick_callback(self.tick_01)
            elif seq == 2:
                unreal.log("*** DEBUG: Registering ::tick_02 callback")
                self.slate_post_tick_handle = \
                    unreal.register_slate_post_tick_callback(self.tick_02)
            self.frame_count = 0

            print("Test")

    def tick_01(self, delta_time: float):
        # Potentially end the thread
        if self.frame_count > self.max_frame_count:
            unreal.log_error("<<<<< Hit max_frame_count. Terminating <<<<<<")
            # End tick and any changes
            unreal.unregister_slate_post_tick_callback(
                self.slate_post_tick_handle)

        #######################################################
        #           Perform Actions at ticks
        #######################################################

        # 1) Reset at end of actions or as a protective measure
        if self.action_count == self.num_actions\
                or self.frame_count > self.max_frame_count - 1:
            # Reset before the end of the thread
            self.teleport_camera_home(home=self.cam_home)
            self.action_count += 1

        # 2) Perform action sequence
        elif self.action_count < self.num_actions:
            self.run_image_seq1(self.action_count)

            self.action_count += 1

        self.frame_count += 1


    def tick_02(self, delta_time: float):
        # Potentially end the thread
        if self.frame_count > self.max_frame_count:
            unreal.log_error("<<<<< Hit max_frame_count. Terminating <<<<<<")
            # End tick and any changes
            unreal.unregister_slate_post_tick_callback(
                self.slate_post_tick_handle)

        #######################################################
        #           Perform Actions at ticks
        #######################################################

        # 1) Reset at end of actions or as a protective measure
        if self.action_count == self.num_actions\
                or self.frame_count > self.max_frame_count - 1:
            # Reset before the end of the thread
            self.teleport_camera_home(home=self.cam_home)
            self.action_count += 1

        # 2) Perform action sequence
        elif self.action_count < self.num_actions:

            if self.frame_count % 3 == 0:
                self.step_camera_rotation_seq()

                self.action_count += 1

        self.frame_count += 1

    #######################################################
    #           Test Sequences
    #######################################################

    def run_image_seq1(self, action_num):
        '''
        First Proof of Concept demo that highlights how asset textures,
        lighting and camera position can all be sequentially be changed from
        a python script. After every change, a screenshot is taken.

        Changes occur on a semi-separate thread from the UE4 editor every
        processor 'tick'. Every 'tick' a different change occurs so that a
        screenshot of every combination of differences can be captured in a
        different image.

        Material param changes: [0, 1] flip of ROUGHNESS param for two assests
        Lighting param changes: Random intensity & color change of two
                                    separate point lights
        Camera movement:    Scalar increase in position to show 'pan'
        :param action_num:
        :return:
        '''
        print(f"---- Performing Action {action_num}")

        # Change the Editor
        #   1. Change Material params
        #   2. Change Material Alpha maps
        #   3. Change lighting intensity and colors
        #   4.
        MODULUS_MOVES = 5
        if action_num % MODULUS_MOVES == 0 or action_num % MODULUS_MOVES == 1:
            # Change the material textures

            # 1) Material attribute params
            self.toggle_param(self.mat0, "rough")
            self.toggle_param(self.mat1, "text1_rough")
            self.toggle_param(self.mat1, "text2_rough")
            self.toggle_param(self.mat2, "AddedOpacity")

        elif action_num % MODULUS_MOVES == 2:
            # Change Material Alpha_Map location (HARD COPIED)
            #       [-70, 40] & [-70, 50] are the max alpha_* offsets
            val1 = random.randrange(-50, 20, 1) / 100
            val2 = random.randrange(-50, 30, 1) / 100
            self.set_param(self.mat1, "alpha_x", val1)
            self.set_param(self.mat1, "alpha_y", val2)

        elif action_num % MODULUS_MOVES == 3:
            # Toggle lighting
            comp1 = self.point_light1.get_component_by_class(
                                            unreal.PointLightComponent)
            c1 = [random.randint(0, 255), random.randint(0, 255),
                  random.randint(0, 255)]
            i1 = random.randint(0, 40)
            comp1.set_editor_property("Intensity", i1)
            comp1.set_editor_property("light_color", c1)
            print("*** Changed PointLight_1")

            comp2 = self.point_light2.get_component_by_class(
                                            unreal.PointLightComponent)
            c2 = [random.randint(0, 255), random.randint(0, 255),
                  random.randint(0, 255)]
            i2 = random.randint(0, 40)
            comp2.set_editor_property("Intensity", i2)
            comp2.set_editor_property("light_color", c2)
            print("*** Changed PointLight_2")

        else:
            # Move Camera
            self.step_camera_rotation_seq()

        # Take a screenshot
        self.take_HighResShot2()

    def step_camera_rotation_seq(self):
        '''
        Second Proof of Concept demo demonstrating intentionally rotating the
        camera in a spherical orbit around an object of Importance. Every
        position will take a screenshot.

        User defines the locations by giving a list of spherical coordinates
        (rho, alpha, phi) that the camera should go too.
        '''

        if len(self.ss_cam_locs) == 0 or len(self.ss_cam_angles) == 0:
            raise Exception("ERROR: Need to calculate camera rotation angels.")

        seq_pos = self.ss_cur_cam_position

        (f"---- Performing Action {seq_pos}")

        try:
            unreal.log(f"*** Rotation seq #{seq_pos}")
            cam_loc = self.ss_cam_locs[seq_pos]
            cam_rot = self.ss_cam_angles[seq_pos]

            unreal.log(f"*** DEBUG: location = "
                       f"{cam_loc[0]}/{cam_loc[1]}/{cam_loc[2]}")
            unreal.log(f"*** DEBUG: angle = {cam_rot}")
            self.teleport_camera(_x=cam_loc[0],
                                 _y=cam_loc[1],
                                 _z=cam_loc[2])
            self.rotate_camera(roll=cam_rot["roll"],
                               pitch=cam_rot["pitch"],
                               yaw=cam_rot["yaw"])

            # # Take a screenshot
            # self.take_HighResShot2()

            self.ss_cur_cam_position = self.ss_cur_cam_position + 1

        except IndexError:
            unreal.log_warning(f"*** WARN: Attempting too many camera "
                              f"rotations. "
                  f"Only {len(self.ss_cam_locs)} camera locations are "
                  f"specified.")

    #######################################################
    #           CineCameraActor Methods
    #######################################################

    def take_HighResShot2(self):
        unreal.log("***** Take screenshot of current Editor")
        unreal.SystemLibrary.execute_console_command(
            self.editor.get_editor_world(), "HighResShot 2")


    def teleport_camera(self, _x, _y, _z):
        # Move first
        self.cam_scene.set_relative_location(new_location=[_x, _y, _z],
                                          sweep=False,
                                          teleport=True)
        self.cam_scene.set_relative_rotation(new_rotation=[0, 0, 0],
                                             sweep=False,
                                             teleport=True)
        current_loc = self.cam_scene.relative_location
        unreal.log_warning(f"ALTERED Camera location: {current_loc}")

    def teleport_camera_home(self, home):
        unreal.log_warning("*** Return camera to HOME position ***")
        self.teleport_camera(_x=home.x,
                             _y=home.y,
                             _z=home.z)

    def rotate_camera(self, roll, pitch, yaw):

        pitch = pitch - 90
        yaw = yaw - 180

        rot = unreal.Rotator(roll=roll, pitch=pitch, yaw=yaw)
        self.cam_scene.set_relative_rotation(new_rotation=rot,
                                                  sweep=False,
                                                  teleport=True)
        current_rot = self.cam_scene.relative_rotation
        unreal.log_warning(f"ALTERED releative Camera rotation: {current_rot}")

    #######################################################
    #           Change Material Parameter Methods
    #######################################################

    def setup_material(self, material_path):
        print("*** Loading material")

        asset = unreal.load_object(None, material_path)

        mat_name = asset.get_name()
        mat_parent = asset.get_class()
        mat_params = asset.scalar_parameter_values

        print(f"Material Details:\n\t{mat_name}"
              f"\n\t{mat_parent}"
              f"\n\t{mat_params}")

        return asset

    def get_material_params(self, mat):
        return mat.get_scalar_parameter_value

    def toggle_param(self, mat, param: str):
        # Toggle "rough" parameter using MaterialEditingLibrary() methods
        editor = unreal.MaterialEditingLibrary()
        val = editor.get_material_instance_scalar_parameter_value(mat, param)
        val = 0 if val else 1
        editor.set_material_instance_scalar_parameter_value(mat, param, val)

    def set_param(self, mat, param: str, val: float):
        # Set material parameter using MaterialEditingLibrary() methods
        editor = unreal.MaterialEditingLibrary()
        c_val = editor.get_material_instance_scalar_parameter_value(mat, param)
        editor.set_material_instance_scalar_parameter_value(mat, param, val)

    #######################################################
    #           Move Camera Methods
    #######################################################

    def calc_cam_rotation_track(self,
                                range_rho: list,
                                range_alpha: list,
                                range_phi: list):
        end_cam_locs = []
        end_cam_angles = []

        for r in range_rho:
            for phi in range_phi:
                for alpha in range_alpha:
                    x, y, z = spherical_to_cartisen_deg(rho=r,
                                                        alpha=alpha,
                                                        phi=phi)
                    end_cam_locs.append([x, y, z])
                    end_cam_angles.append({'roll': 0.0,
                                           'pitch': phi,
                                           'yaw': alpha})

        if len(end_cam_locs) > 0:
            unreal.log_warning("*** DEBUG: Camera rotation locations set.")
            self.ss_cam_locs = end_cam_locs
        else:
            unreal.log_error("*** WARN: Camera rotations not set!!!!")
            self.ss_cam_locs = None

        if len(end_cam_angles) > 0:
            unreal.log_warning("*** DEBUG: Camera rotation angles set.")
            self.ss_cam_angles = end_cam_angles
        else:
            unreal.log_error("*** WARN: Camera rotations not set!!!!")
            self.ss_cam_angles = None


#######################################################
#           Start of MAIN()
#######################################################

home_path = '/Game/_MENG_Project/00_Proof_of_Concept/'
material_paths = [
    home_path + 'metal_p_blue1.metal_p_blue1',
    # home_path + 'metal_p_rust1_Inst1.metal_p_rust1_Inst1'
    home_path + 'metal_p_rust_STAR_alpha_Inst1.metal_p_rust_STAR_alpha_Inst1',
    home_path + 'rust_SQ_alpha_inst1.rust_SQ_alpha_inst1'
    ]

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
        print("*** DEBUG: Found Camera actor")

    if y.get_name() == "PointLight_1":
        p_light1 = y
        print("*** DEBUG: Found PointLight_1 actor")

    if y.get_name() == "PointLight_2":
        p_light2 = y
        print("*** DEBUG: Found PointLight_2 actor")

    if y.get_name() == "Cube_cam_obj":
        poi_obj = y
        unreal.log("*** DEBUG: Found Cube_cam_obj actor")

actor_dic = {"level_editor": editor,
                "camera": cam,
                "point_of_interest_obj": poi_obj,
                "point_light1": p_light1,
                "point_light2": p_light2}

cam_home = [-200, 0, 200]

test1 = MoveCamera(60, material_paths, actor_dic, cam_home)


#####
#   Max angles for this demo
#       Rho = inf
#       alpha = [60, 300]
#       phi = [0, 90]
range_rho = [400]
range_alpha = [90, 180, 270]
range_phi = [30, 45, 60, 90]

test1.calc_cam_rotation_track(range_rho=range_rho,
                              range_alpha=range_alpha,
                              range_phi=range_phi)

test1.start(1)
#test1.start(2)





