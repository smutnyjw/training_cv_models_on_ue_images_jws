'''
File:   00-main-CatDog_ImagePipeline.py
Author: John Smutny
Date:   03/10/2024
Description: 
    Singular file that is executed in an Unreal Engine 4.27 script terminal.
    This script will create a class object that will perform various actions
    to an UE Editor on the same thread as the UE Editor window and collect
    screenshots. This script requires that various UE Assets be already
    present in the Editor environment named 'Cat' or 'Dog' in the .uasset name.

    The script will collect screenshots of various UE assets from various
    CineCameraActor angles and output a .csv file of all images created as
    well as what object is in the image based on the .uasset's filename.

    cmd:    python "< insert path to file>/04_ImagePipeline_01b_onefile.py"

Other Notes:

    Please enter all user inputs at the bottom of this script. Such as...
    - Desired camera angles
    - Desired output path for the collected screenshots (output_path_to_images)
    - Absolute file paths for all textures used in the collection.

'''

import os
from datetime import datetime

from math import pi, sin, cos

import unreal

#######################################################
#######################################################
#######################################################

def link_ue_objects():
    LOC_POINTS = []
    cat_actor_objs = []
    dog_actor_objs = []
    for y in actors:
        name = y.get_name()

        if name == "Main_Camera":
            # Ensure that the camera Actor is selected for screenshots
            cam = y
            print("*** DEBUG: Found Camera actor")

        if 'CENTER_POINT' == name:
            CENTER_POINT = y
            print("*** DEBUG: Found CENTER_POINT of the shot")

        if 'LOC' in name:
            LOC_POINTS.append(y)
            print(f"*** DEBUG: Found actor location {name}")

        if name == 'Sphere_Brush_StaticMesh':
            sphere = y
            print("*** DEBUG: Found Sphere object")

        if name == 'Floor':
            floor = y
            print("*** DEBUG: Found Floor object")

        if 'Cat' in name or 'cat' in name:
            cat_actor_objs.append(y)
            print(f"*** DEBUG: Found Cat actor {name}")

        if 'Dog' in name:
            dog_actor_objs.append(y)
            print(f"*** DEBUG: Found Dog actor {name}")

        if 'SunSky' in name:
            sunsky = y
            print(f"*** DEBUG: Found the SunSky actor")

    print(f"Cats = {len(cat_actor_objs)}\tDogs = {len(dog_actor_objs)}")

    poi_objs = cat_actor_objs + dog_actor_objs

    actor_dic = {"level_editor": unreal.EditorLevelLibrary(),
                 "camera": cam,
                 "sunsky": sunsky,
                 "center_point": CENTER_POINT,
                 "model_locs": LOC_POINTS,
                 "points_of_interest_obj": poi_objs,
                 "sphere": sphere,
                 "floor": floor
                 }

    return actor_dic


def check_user_inputted_UE_assets(material_paths,
                                  material_params,
                                  scene_presets):
    for i in range(len(material_paths)):
        obj = unreal.load_object(None, material_paths[i])
        assert obj

        mat_params = obj.get_editor_property("scalar_parameter_values")
        mat_param_list = []
        for i in range(len(mat_params)):
            info = mat_params[i].get_editor_property("parameter_info")
            mat_param_list.append(str(info.get_editor_property("name")))

        for param in material_params:
            if param not in mat_param_list:
                raise Exception("*** ERROR: Double check specified Material "
                                "Instance. Did not find needed MATERIAL_PARAM "
                                f"{param}. Make sure {param} is in the Material "
                                "Instance .uasset file.")

    for key in scene_presets.keys():
        switches = scene_presets[key]
        if len(switches) != len(material_params):
            raise Exception("*** Error: update PRESET settings. Unequal "
                            f"dimension for setting index {key}")

#######################################################

def spherical_to_cartisen_deg(rho: int = 0.0,
                              alpha: float = 0.0,
                              phi: float = 0.0):
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

#######################################################
#######################################################
#######################################################

class MoveCamera(object):
    def __init__(self, ue_objs: dict, cam_home, output_path_of_run: str):
        # Threading setup
        self.in_progress = False
        self.frame_count = 0
        self.frame_buffer = -1
        self.max_frame_count = -1
        self.action_count = 0
        self.num_actions = -1

        self.slate_post_tick_handle = None

        # Definition of the class's state-machine
        self._state = {"set_new_model": True,
                       "set_new_scene": False,
                       "get_snapshots": False,
                       "shift_to_new_model": False,
                       "end_of_models_terminate_early": False}

        ###########
        # UE Editor controls
        self.editor = ue_objs["level_editor"]

        # Camera setup
        self.camera_obj = ue_objs["camera"]
        self.cam_scene = self.camera_obj.scene_component
        self.cam_home = unreal.Vector(x=cam_home[0],
                                      y=cam_home[1],
                                      z=cam_home[2])

        # SunSky actor setup
        #   Properties: TimeZone, SolarTime
        self.sunsky = ue_objs["sunsky"]
        self.sun_time = None

        # Point of Interest for the camera to focus on
        self.poi_loc_cp = ue_objs["center_point"]
        self.poi_loc_ss = ue_objs["model_locs"]
        self.poi_objs = ue_objs["points_of_interest_obj"]

        # Scene components
        self.material_paths = None
        self.material_params = None
        self.scene_presets = None

        ##############
        # Output and Debug information
        self.debug_current_poi = 0
        self.debug_current_angle = 0
        self.debug_current_scene = 0
        self.debug_current_sun = 0

        self.output_path = output_path_of_run

        self.log_img_str = "abs_filepath,class\n"

    #######################################################
    #           Setter & Getter methods
    #######################################################

    def get_status(self):
        return self.in_progress

    def set_max_frames(self, num_frames):
        self.max_frame_count = num_frames

    def set_frame_buffer(self, num):
        self.frame_buffer = num

    def set_num_actions(self, num):
        self.num_actions = num

    def set_material_paths(self, paths):
        self.material_paths = paths

    def set_material_params(self, params):
        self.material_params = params

    def set_scene_presets(self, presets):
        self.scene_presets = presets

    def set_scene_sun_north_offset(self, north_offset):
        self.sun_time = north_offset

    #######################################################
    #           Methods for UE environment setup
    #######################################################

    def reset_model_locations(self, loc):

        dest = loc

        start_x = dest.static_mesh_component.relative_location.x
        SPACING = 50

        for poi in self.poi_objs:
            dest_vect = unreal.Vector(
                x=start_x,
                y=dest.static_mesh_component.relative_location.y,
                z=dest.static_mesh_component.relative_location.z)
            poi.static_mesh_component.set_editor_property(
                                                "relative_location",
                                                dest_vect)
            start_x = start_x + SPACING
        unreal.log("*** DEBUG - Reset all model locations to START")

    def set_environment(self):
        '''
        Method that will set the UE environment to a default state.
        Specifically
        1. setting the camera to the stored HOME coordinates
        2. removing camera object tracking
        3. resetting sunsky setting
        4. resetting all UE models to the START location
        5. resetting the internal class state-machine
        6. resetting the various debug counts.
        :return:
        '''
        unreal.log("::set_environment()")

        # Set initial scene
        self.set_scene(self.material_paths,
                       self.material_params,
                       self.debug_current_scene)

        # Set initial camera location
        self.teleport_camera_home(home=self.cam_home)
        print(f"HOME Camera location: {self.cam_home}")

        track_set = unreal.CameraLookatTrackingSettings(
                                enable_look_at_tracking=False)
        self.camera_obj.set_editor_property("lookat_tracking_settings",
                                            track_set)

        # Set the sun lighting
        solar_time = self.sun_time[0]
        self.sunsky.set_editor_property("SolarTime", 15.0)
        self.sunsky.set_editor_property("NorthOffset", solar_time)

        # Reset the location of all of the models
        self.reset_model_locations(self.poi_loc_ss[0])

        # Reset class counts
        self.frame_count = 0
        self.action_count = 0

        self._state["set_new_model"] = True
        self._state["set_new_sun_time"] = False
        self._state["set_new_scene"] = False
        self._state["get_snapshots"] = False
        self._state["shift_to_new_model"] = False
        self._state["end_of_models_terminate_early"] = False
        self.debug_current_poi = 0
        self.debug_current_angle = 0
        self.debug_current_scene = 0
        self.debug_current_sun = 0

    #######################################################
    #           Test Sequences
    #######################################################

    def start(self, seq: int):
        '''
        Trigger to begin the screenshot sequence. If the class is set
        correctly then a 'tick_callback' interrupt will be created to be
        executed on the same thread as the UE Editor (this is on the SAME
        thread, not a parallel thread).
        :param seq:     User specified input as too which sequence to start.
        :return:
        '''
        self.editor.clear_actor_selection_set()
        self.editor.set_selected_level_actors([self.camera_obj])

        if self.max_frame_count == -1:
            raise Exception("ERROR: please set the maximum allowed frame "
                            "count of this script. Run the MoveCamera class "
                            "method ... set_max_frames()")

        if self.frame_buffer == -1:
            raise Exception("ERROR: please set the buffer between actions. "
                            "The frame buffer allows for adequete time for "
                            "the UE Editor to update before performing the "
                            "next script action. If not enough time is "
                            "allowed, actions will be skipped. Please run the "
                            "MoveCamera class method ... set_frame_buffer()"
                            "\n- A good buffer could start at 30")

        if self.num_actions == -1:
            raise Exception("ERROR: please set the number of expected actions "
                            "to be performed by this script. Please run the "
                            "MoveCamera class method ... set_num_actions()")

        if self.frame_count != 0:
            print("Please wait until first call is done.")
        else:
            # Create output for run
            self.create_output_dir(self.output_path)

            if seq == 1:
                unreal.log("*** DEBUG: Registering ::tick_01 callback")
                self.frame_count = 0
                self.slate_post_tick_handle = \
                    unreal.register_slate_post_tick_callback(self.tick_01)
                self.in_progress = True

    def tick_01(self, delta_time: float):
        '''
        This method runs every UE 'tick_callback'. Every tick, the script
        checks if the callback should stop (the end of an action sequence) or
        continue.
        :param delta_time: Not used
        :return: NA
        '''
        # Potentially end the thread
        terminate = False
        if self.frame_count >= self.max_frame_count:
            unreal.log_error("<<<<< Hit max_frame_count. Terminating <<<<<<")
            terminate = True

        elif self.action_count >= self.num_actions:
            unreal.log_error("<<<<< Hit max_num_actions. Terminating <<<<<<")
            terminate = True

        #######################################################
        #           Perform Actions at ticks
        #######################################################

        # 1) Reset at end of actions or as a protective measure
        if terminate:
            self.set_environment()

            self.output_image_csv(path=f"{self.output_path}/{self.output_dir}",
                                  filename="synth-cats_dogs.csv")

            # Reset before the end of the thread
            self.teleport_camera_home(home=self.cam_home)
            self.action_count += 1
            unreal.log("Unregister callback")
            unreal.unregister_slate_post_tick_callback(
                self.slate_post_tick_handle)

            self.in_progress = False

        # 2) Perform action sequence
        elif self.action_count < self.num_actions:

            # Only execute actions every 'frame_buffer' frames
            if self.frame_count % self.frame_buffer == 0:
                try:
                    terminate = self.run_image_seq1(self.action_count)
                except Exception:
                    unreal.log_error("CAUGHT ERROR: "
                                     f"poi {self.debug_current_poi}, "
                                     f"cam {self.debug_current_angle}, "
                                     f"scene {self.debug_current_scene}, "
                                     f"sun {self.debug_current_sun}")
                    unreal.log("Unregister callback")
                    unreal.unregister_slate_post_tick_callback(
                        self.slate_post_tick_handle)
                    terminate = True

                if terminate:
                    self.action_count = self.num_actions

        self.frame_count += 1

    def run_image_seq1(self, screenshot_num):
        '''
        This is the main action sequence in the image gathering script.
        While the script meets all non-terminate conditions, the script will
        cycle through the established class state-machine. Thus iterating
        through all available options of model, sunsky lighting,
        scene preset, and camera angle. After every change, a screenshot is
        taken.

        Changes occur on a semi-separate thread from the UE4 editor every
        processor 'tick'. Every 'tick' a different change occurs so that a
        screenshot of every combination of differences can be captured in a
        different image.

        General State-Machine process
        *** Screenshots of all specified angles are taken after each step
        1. Move new object to the center of the background stage
        2. Change skysun lighting
        3. Move object to the END POINT. Check if gone through all models
            - If Yes, then change scene preset and repeat step 1.
            - If no, then begin at step 1.

        :param screenshot_num:  used to keep track of how many screen shots
                                have already been taken
        :return: NA - however, several .png screenshots are taken
        '''
        print(f"---- Performing Action {screenshot_num}")
        print(f"*** DEBUG: debug #s: {self.debug_current_poi}/"
              f"{self.debug_current_angle}/{self.debug_current_scene}/"
              f"{self.debug_current_sun}")

        end_of_models_terminate_early = False

        if self._state["set_new_model"]:
            # Move a new object to the center of the stage

            print("*** STATE: set_new_model")

            # Move new model to CENTER_POINT
            poi = self.poi_objs[self.debug_current_poi]
            self.set_obj_loc(poi, self.poi_loc_cp)

            # Set camera to focus on the target
            self.set_camera_tracking(poi)

            self._state["set_new_model"] = False
            self._state["get_snapshots"] = True

        elif self._state["get_snapshots"]:
            # Collect all images specified in angle inputs

            # Move Camera
            self.step_camera_rotation_seq()

            # Log the image
            image_name, label = self.create_image_name(
                                          action_num=screenshot_num,
                                          model_num=self.debug_current_poi,
                                          scene_num=self.debug_current_scene,
                                          lighting_num=self.debug_current_sun,
                                          angle_num=self.debug_current_angle)
            abs_filepath = f"{self.output_path}/{self.output_dir}/" \
                           f"images/{image_name}"
            self.log_image(abs_filepath=abs_filepath, label=label)

            # Take a screenshot
            self.take_HighResShot2(abs_filepath=abs_filepath)
            self.action_count += 1

            if self.debug_current_angle == len(self.ss_cam_locs):
                self.debug_current_angle = 0
                self._state["get_snapshots"] = False
                self._state["set_new_sun_time"] = True

        elif self._state["set_new_sun_time"]:
            # Change the SunSky lighting

            print("*** STATE: set_new_sun_time")

            # Iterate the current sun time setting to cycle scene lighting
            self.debug_current_sun = self.debug_current_sun + 1

            # Only iterate if a setting has not been executed
            if self.debug_current_sun < len(self.sun_time):
                solar_time = self.sun_time[self.debug_current_sun]
                self.sunsky.set_editor_property("NorthOffset", solar_time)

                self._state["set_new_sun_time"] = False
                self._state["get_snapshots"] = True

            else:
                self.debug_current_sun = 0
                solar_time = self.sun_time[self.debug_current_sun]
                self.sunsky.set_editor_property("NorthOffset", solar_time)

                self._state["set_new_sun_time"] = False
                self._state["shift_to_new_model"] = True

        elif self._state["set_new_scene"]:
            # Change the preset texture multiplexers to cycle to the next
            # preset background, floor, and scene

            unreal.log_warning("*** STATE: set_new_scene")

            self.debug_current_scene = self.debug_current_scene + 1
            self.set_scene(self.material_paths,
                           self.material_params,
                           self.debug_current_scene)

            self._state["set_new_scene"] = False
            self._state["set_new_model"] = True

        elif self._state["shift_to_new_model"]:
            # Cycle the model used at the center of the stage.

            print("*** STATE: shift_to_new_model")

            # Decide next action based on how many models are used
            #   1) Gone through all models, terminate sequence
            #   2) or Move onto the next model, repeat sequence
            #   3) or Set a new scene, repeat sequence from first model
            if self.debug_current_poi == len(self.poi_objs)\
                    and self.debug_current_scene == len(self.scene_presets):
                self._state["shift_to_new_model"] = False
                self._state["end_of_models_terminate_early"] = True
            elif self.debug_current_poi < len(self.poi_objs)-1:
                # Move current model out of the CENTER POINT
                self.set_obj_loc(self.poi_objs[self.debug_current_poi],
                                 self.poi_loc_ss[1])

                self.debug_current_poi = self.debug_current_poi + 1

                self._state["shift_to_new_model"] = False
                self._state["set_new_model"] = True

            else:
                print("*** DEBUG: attempt to set new scene")

                # Move current model out of the CENTER POINT
                self.set_obj_loc(self.poi_objs[-1],
                                 self.poi_loc_ss[1])

                self.debug_current_poi = 0

                # Change Scene
                self._state["shift_to_new_model"] = False
                self._state["set_new_scene"] = True

        elif self._state["end_of_models_terminate_early"]:
            end_of_models_terminate_early = True

        return end_of_models_terminate_early

    #######################################################
    #           CineCameraActor Methods
    #######################################################

    def take_HighResShot2(self, abs_filepath):
        cmd = f"HighResShot 2 filename=\"{abs_filepath}\""
        unreal.SystemLibrary.execute_console_command(
            self.editor.get_editor_world(), cmd)

    def teleport_camera(self, _x, _y, _z):
        '''
        Move the camera to the desired cartesian [x, y, z] location.
        '''
        self.cam_scene.set_relative_location(new_location=[_x, _y, _z],
                                          sweep=False,
                                          teleport=True)
        self.cam_scene.set_relative_rotation(new_rotation=[0, 0, 0],
                                             sweep=False,
                                             teleport=True)
        # REF: current_loc = self.cam_scene.relative_location

    def teleport_camera_home(self, home):
        self.teleport_camera(_x=home.x,
                             _y=home.y,
                             _z=home.z)

    def set_camera_tracking(self, poi_obj):
        '''
        Set the camera actor to focus at the center point of a particular UE
        asset.
        :param poi_obj: UE asset that the camera should follow
        :return: NA
        '''
        track_set = unreal.CameraLookatTrackingSettings(
                                enable_look_at_tracking=True,
                                actor_to_track=poi_obj,
                                relative_offset=unreal.Vector(z=50.0))
        self.camera_obj.set_editor_property("lookat_tracking_settings",
                                            track_set)

    def rotate_camera(self, roll, pitch, yaw):
        '''
        Rotate the camera as desired if the CameraLookatTrackingSettings is
        not enabled. Method will not function if CameraLookatTrackingSettings
        is enabled, instead the camera's pitch and yaw will adjust to the
        tracked target
        (For most cases, there will be 0.0 roll)
        '''
        pitch = pitch
        yaw = yaw

        rot = unreal.Rotator(roll=roll, pitch=pitch, yaw=yaw)
        self.cam_scene.set_relative_rotation(new_rotation=rot,
                                                  sweep=False,
                                                  teleport=True)
        # REF: current_rot = self.cam_scene.relative_rotation

    def step_camera_rotation_seq(self):
        '''
        Move the camera to the next cartesian location calculated by
        the ::calc_cam_cartesian_locs() method.
        Rotates the camera in a spherical orbit around an object of Importance.
        Every position will take a screenshot.

        User defines the locations by giving a list of spherical coordinates
        (rho, alpha, phi) that the camera should go too.
        '''

        if len(self.ss_cam_locs) == 0 or len(self.ss_cam_angles) == 0:
            raise Exception("ERROR: Need to calculate camera rotation angels.")

        seq_pos = self.debug_current_angle

        print(f"---- Performing Camera Action {seq_pos}")

        try:
            unreal.log(f"*** Rotation seq #{seq_pos}")
            cam_loc = self.ss_cam_locs[seq_pos]
            cam_rot = self.ss_cam_angles[seq_pos]

            # unreal.log(f"*** DEBUG: location = "
            #            f"{cam_loc[0]}/{cam_loc[1]}/{cam_loc[2]}")
            # unreal.log(f"*** DEBUG: angle = {cam_rot}")
            self.teleport_camera(_x=cam_loc[0],
                                 _y=cam_loc[1],
                                 _z=cam_loc[2])
            self.rotate_camera(roll=cam_rot["roll"],
                               pitch=cam_rot["pitch"],
                               yaw=cam_rot["yaw"])

            # # Take a screenshot
            # self.take_HighResShot2()

            self.debug_current_angle = self.debug_current_angle + 1

        except IndexError:
            unreal.log_warning(f"*** WARN: Attempting too many camera "
                              f"rotations {self.debug_current_angle}. "
                  f"Only {len(self.ss_cam_locs)} camera locations are "
                  f"specified.")

    def calc_cam_cartesian_locs(self,
                                range_rho: list,
                                range_alpha: list,
                                range_phi: list):
        '''
        Method that will calculate the equivalent cartesian coordinates for
        the various combinations of camera angles desired (that are in
        spherical coordinates).
        :param range_rho:   list of radial distances the camera should be
                            from the object
        :param range_alpha: list of xy axis angles the camera should be
        :param range_phi:   list of z-axis angles the camera should be
        :return: sets a class list of camera locations
        '''
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
            unreal.log_warning("::calc_cam_cartesian_locs() - Camera "
                               "rotation locations set.")
            self.ss_cam_locs = end_cam_locs
        else:
            unreal.log_error("::calc_cam_cartesian_locs() - WARN: Camera "
                             "rotations not set!!!!")
            self.ss_cam_locs = None

        if len(end_cam_angles) > 0:
            unreal.log_warning("::calc_cam_cartesian_locs() - Camera "
                               "rotation angles set.")
            self.ss_cam_angles = end_cam_angles
        else:
            unreal.log_error("::calc_cam_cartesian_locs() - Camera "
                             "rotations not set!!!!")
            self.ss_cam_angles = None


    #######################################################
    #           Object Actor Methods
    #######################################################

    def set_obj_loc(self, poi_obj, new_loc):

        dest_vect = unreal.Vector(
            x=new_loc.static_mesh_component.relative_location.x,
            y=new_loc.static_mesh_component.relative_location.y,
            z=new_loc.static_mesh_component.relative_location.z)
        poi_obj.static_mesh_component.set_editor_property("relative_location",
                                                        dest_vect)

    #######################################################
    #           Object Material Methods
    #######################################################

    def set_scene(self, material_paths, material_params, scene_num: int):
        # Get the scene settings
        key = list(self.scene_presets.keys())[scene_num]

        print(f"key: {key} --- num: {scene_num}")

        try:
            settings = self.scene_presets[key]
        except Exception:
            unreal.log_error("--- ERROR: Expecting settings to have key "
                             f"{key}")
            return 0

        try:
            for i in range(len(material_paths)):
                for j in range(len(material_params)):
                    # Set scene
                    material = material_paths[i]
                    param = material_params[j]
                    self.set_param(material, param, settings[j])

            self.sunsky.set_editor_property("SolarTime",
                                    self.sun_lighting[self.debug_current_sun])
        except Exception:
            unreal.log_error("--- ERROR: Tried accessing an undeclared "
                             f"scene_preset. Current settings = {settings}")
            return 0

    def set_param(self, mat_path, param_name: str, val: float):
        obj = unreal.load_object(None, mat_path)
        assert obj

        # Set material parameter using MaterialEditingLibrary() methods
        editor = unreal.MaterialEditingLibrary()

        editor.set_material_instance_scalar_parameter_value(obj, param_name, val)

    #######################################################
    #           log images
    #######################################################

    def create_output_dir(self, path):
        now = datetime.now()
        start = now.strftime("%Y_%m_%d-%H_%M_%S")
        new_dir = start+"-pipeline"

        try:
            os.mkdir(f"{path}/{new_dir}")
            self.output_dir = new_dir
        except OSError as error:
            print(error)

    def create_image_name(self, action_num,
                          model_num, scene_num, lighting_num, angle_num):
        '''
        Method that will design a string for each unique screenshot name.
        :return: str - the unique image name
        '''
        try:
            label = "cat" if "Cat" in self.poi_objs[model_num].get_name() \
                            else "dog"
            debug_msg = f"{scene_num}_{model_num}_" \
                        f"{lighting_num}_{angle_num}"
        except Exception:
            unreal.log_error("*** ERROR: issue in ::create_image_name().")

        image_name = f"{label}_{action_num}-{debug_msg}.png"

        return image_name, label

    def log_image(self, abs_filepath, label):
        '''
        Method that will append a string that records all images taken via a
        pair {image, label of image}
        :param abs_filepath:    full path of where the image is
        :param label:   what is in the image
        :return: NA
        '''
        self.log_img_str = self.log_img_str + f"{abs_filepath},{label}\n"

    def output_image_csv(self, path, filename):
        '''
        Method that will create .csv file log of all taken image screenshots.
        :param path: location of the eventual csv
        :param filename: name of the eventual csv
        :return: creates a csv at the specified path+filename of all taken
                    images.
        '''
        try:
            f = open(f"{path}/{filename}", "w")
            f.write(self.log_img_str)
            f.close()
            unreal.log_warning(f"Write csv file to {path}")

        except NameError:
            print(f"**** WARN: Dataframes to {path} was not defined")


#######################################################
#######################################################
#######################################################

if __name__=="__main__":

    ############################
    #   User defined Unreal Assets

    HOME_PATH = '/Game/MENG_POC_04_ImagePipeline01/TextureSelect/'
    MATERIAL_PATHS = [
        HOME_PATH + 'background_select_Mat_Inst.background_select_Mat_Inst',
        HOME_PATH + 'scene_select_Mat_Inst.scene_select_Mat_Inst',
        HOME_PATH + 'floor_select_Mat_Inst.floor_select_Mat_Inst'
        ]

    SCENE_PRESETS = {
        # Multiplexer decoding of material instance parameters to select a texture.
        #   Switch values are based on LERP nodes, not switches.
        #   Values must be 0.0 or 1.0
        "leaves": [0, 0, 0, 0, 0],
        "forest": [1, 0, 0, 0, 0],
        "interior01": [0, 0, 1, 0, 0],
        "grassland": [0, 1, 1, 0, 0],
        "interior02": [0, 0, 0, 0, 1],
        "mountain": [0, 0, 0, 1, 1]

    }
    MATERIAL_PARAMS = ['sw_1a', 'sw_1b', 'sw_2', 'sw_1c', 'sw_3']

    # Checks
    check_user_inputted_UE_assets(MATERIAL_PATHS,
                                  MATERIAL_PARAMS,
                                  SCENE_PRESETS)

    ############################
    #   Operational settings

    output_path_to_images = "..\\_data\\cat_dog\\synth"

    CAM_HOME = [0, -300, 200]
    FRAME_BUFFER = 30

    #####
    #   Valid settings for image collection (in Degrees)
    #       SUN_NORTH_OFFSET =  [ 75 --> 190 ]
    #       Rho =               [ 100 --> 400 ]
    #       alpha =             [ -180 --> 0 ]
    #       phi =               [ 0 --> 90 ]
    #
    #   NOTE:
    #   Unlit mode
    #   https://docs.unrealengine.com/4.27/en-US/PythonAPI/class/ViewModeIndex.html?highlight=unlit
    SUN_NORTH_OFFSET = [ 190 ]
    RANGE_RHO = [200, 240, 290 ]
    RANGE_ALPHA = [-140, -105, -95, -85, -75, -55]
    RANGE_PHI = [45, 55, 65, 80]


    #######################################################
    #######################################################
    #######################################################

    #######################################################
    #           Start of MAIN()
    #######################################################

    home_path = HOME_PATH
    material_paths = MATERIAL_PATHS
    material_params = MATERIAL_PARAMS

    #####
    # Code to print out all or specific actors in a loaded level
    actors = unreal.EditorLevelLibrary().get_all_level_actors()

    actor_dic = link_ue_objects()

    cam_home = CAM_HOME
    test1 = MoveCamera(ue_objs=actor_dic,
                       cam_home=cam_home,
                       output_path_of_run=output_path_to_images)

    ####################################################
    #   Define test parameters
    ####################################################

    range_rho = RANGE_RHO
    range_alpha = RANGE_ALPHA
    range_phi = RANGE_PHI

    test1.calc_cam_cartesian_locs(range_rho=range_rho,
                                  range_alpha=range_alpha,
                                  range_phi=range_phi)

    num_models = len(actor_dic["points_of_interest_obj"])
    num_angles = len(range_rho)*len(range_alpha)*len(range_phi)
    num_actions = num_models * num_angles * len(SCENE_PRESETS.keys()) * \
                  len(SUN_NORTH_OFFSET)

    num_frames = num_actions * FRAME_BUFFER * 5.0

    test1.set_max_frames(num_frames)
    test1.set_frame_buffer(FRAME_BUFFER)
    test1.set_num_actions(num_actions)
    test1.set_material_paths(material_paths)
    test1.set_material_params(material_params)
    test1.set_scene_presets(SCENE_PRESETS)
    test1.set_scene_sun_north_offset(SUN_NORTH_OFFSET)

    ####################################################
    #   Execute test for each preset
    ####################################################

    unreal.log_warning("----- START TEST ------")
    test1.set_environment()
    test1.start(1)
    unreal.log_warning("*** DEBUG: PRESET COMPLETE!!!")

