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

'''

import unreal
import random

class MoveCamera(object):
    def __init__(self, iterations, material_paths: list, obj: dict):
        # Threading setup
        self.frame_count = 0
        self.max_frame_count = 50
        self.action_count = 0
        self.num_actions = iterations

        self.slate_post_tick_handle = None

        self.editor = obj["level_editor"]

        # Camera setup
        self.camera = obj["camera"]
        self.cam_scene = self.camera.scene_component
        self.cam_home = unreal.Vector(x=0, y=0, z=0)


        # Material_0 setup
        self.mat0 = self.setup_material(material_paths[0])
        self.mat0_params = self.get_material_params(self.mat0)

        # Material_1 setup
        self.mat1 = self.setup_material(material_paths[1])
        self.mat1_params = self.get_material_params(self.mat1)

        # Point Lights
        self.point_light1 = obj["point_light1"]
        self.point_light2 = obj["point_light2"]

        self.set_environment()

    def set_environment(self):
        print("TODO")

        # Set initial camera
        self.cam_home = unreal.Vector(x=-150, y=-60, z=150)
        self.teleport_camera_home(home=self.cam_home)
        print(f"HOME Camera location: {self.cam_home}")

        # Turn off SunSkylight

        # Set initial light intensities & colors





    def start(self):
        self.editor.clear_actor_selection_set()
        self.editor.set_selected_level_actors([self.camera])
        if self.frame_count != 0:
            print("Please wait until first call is done.")
        else:
            self.slate_post_tick_handle = \
                unreal.register_slate_post_tick_callback(self.tick)
            self.frame_count = 0

            print("Test")

    def tick(self, delta_time: float):
        # Potentially end the thread
        if self.frame_count > self.max_frame_count:
            print("<<<<< Hit max_frame_count. Terminating <<<<<<")
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

    #######################################################
    #           Test Sequences
    #######################################################

    def run_image_seq1(self, action_num):
        print(f"---- Performing Action {action_num}")

        # Change the Editor
        num_moves = 5
        if action_num % num_moves == 0 or action_num % num_moves == 1:
            # Change the material textures

            # 1) Material attribute params
            self.toggle_param(self.mat0, "rough")
            self.toggle_param(self.mat1, "text1_rough")
            self.toggle_param(self.mat1, "text2_rough")

        elif action_num % num_moves == 2:
            # Change Material Alpha_Map location (HARD COPIED)
            #       [-70, 40] & [-70, 50] are the max alpha_* offsets
            val1 = random.randrange(-50, 20, 1) / 100
            val2 = random.randrange(-50, 30, 1) / 100
            self.set_param(self.mat1, "alpha_x", val1)
            self.set_param(self.mat1, "alpha_y", val2)

        elif action_num % num_moves == 3:
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
            self.teleport_camera(_x=self.cam_home.x + 0 * self.frame_count,
                                 _y=self.cam_home.y + 0 * self.frame_count,
                                 _z=self.cam_home.z + 5 * self.frame_count)

        # Take a screenshot
        self.take_HighResShot2()

    #######################################################
    #           CineCameraActor Methods
    #######################################################

    def take_HighResShot2(self):
        print("***** Take screenshot of current Editor")
        unreal.SystemLibrary.execute_console_command(
            self.editor.get_editor_world(), "HighResShot 2")


    def teleport_camera(self, _x, _y, _z):
        # Move first
        self.cam_scene.set_world_location(new_location=[_x, _y, _z],
                                             sweep=False,
                                             teleport=True)
        current_loc = self.cam_scene.get_world_location()
        print(f"ALTERED Camera location: {current_loc}")

    def teleport_camera_home(self, home):
        print("*** Return camera to HOME position ***")
        self.teleport_camera(_x=home.x,
                             _y=home.y,
                             _z=home.z)


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
#           Start of MAIN()
#######################################################
home_path = '/Game/_MENG_Project/00_Proof_of_Concept/'
material_paths = [
    home_path + 'metal_p_blue1.metal_p_blue1',
    # home_path + 'metal_p_rust1_Inst1.metal_p_rust1_Inst1'
    home_path + 'metal_p_rust_alpha_Inst1.metal_p_rust_alpha_Inst1'
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

    if y.get_name() == "PointLight_1":
        p_light1 = y

    if y.get_name() == "PointLight_2":
        p_light2 = y

actor_dic = {"level_editor": editor,
                "camera": cam,
                "point_light1": p_light1,
                "point_light2": p_light2}

test1 = MoveCamera(15, material_paths, actor_dic)

test1.start()





