'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''


import os
import random
from datetime import datetime

from math import pi, sin, cos

import unreal


#################################################
#               The central custom class
#################################################

class UE_ImagePipeline(object):
    def __init__(self, ue_objs: dict, reference_corners: list,
                 num_basecolors: int, num_map_sets: int,
                 plate_offsets: list, cam_offsets_per: list,
                 light_offsets_per: list, cam_home: list,
                 material_path: str, material_params_dict: dict,
                 output_path_of_run: str):
        # Threading setup
        self.in_progress = False
        self.frame_count = 0
        self.frame_buffer = -1
        self.max_frame_count = -1
        self.action_count = 0
        self.max_num_actions = -1

        self.slate_post_tick_handle = None

        # Definition of the class's state-machine
        self.seq_list = []
        self._state = {"set_camera_to_corner": True,
                       "set_basecolor": False,
                       "set_map_set": False,
                       "set_plate_offset": False,
                       "get_snapshots": False,
                       "shift_to_new_model": False,
                       "end_of_models_terminate_early": False}

        ###########
        # UE Editor controls
        self.editor = ue_objs["level_editor"]

        # Camera setup
        self.camera_obj = ue_objs["camera"]
        self.cam_home = cam_home[0]
        self.cam_light_obj = ue_objs['pointlight_camera']
        self.cam_light_home = cam_home[1]

        self.cc_tracker_obj = ue_objs["cc_tracker"]

        # Point of Interest for the camera to focus on
        self.poi_objs = ue_objs["points_of_interest_obj"]

        # Texture variations available
        self.num_basecolors = num_basecolors
        self.num_map_sets = num_map_sets
        self.mat_path = material_path
        self.mat_color_sw = material_params_dict['basecolor']
        material_params_dict.pop('basecolor')
        self.mat_map_set_sws = material_params_dict

        # Weld Plate Locations
        self.plate_corner_refs = reference_corners
        self.offsets_plate = plate_offsets

        # Additional offsets to particular 'plate_offset[]' index to improve
        # image variation.
        # These values are ANGLE VARIATION
        self.offsets_cam = cam_offsets_per
        self.offsets_light = light_offsets_per

        ##############
        # Debug information to control the state machine
        self.debug_current_plate = 0
        self.debug_current_basecolor = 0
        self.debug_current_map_set = 0
        self.debug_current_plate_offset = 0
        self.debug_current_cam_angle = 0
        self.debug_current_lighting = 0

        ##############
        # Output file information
        self.output_path = output_path_of_run
        self.defect_map_sets = [2, 3, 4]
        self.log_df = "abs_filename,class\n"

    #######################################################
    #           Setter & Getter methods
    #######################################################

    def set_max_frames(self, val: int):
        self.max_frame_count = val

    def set_frame_buffer(self, val: int):
        self.frame_buffer = val

    def set_max_num_actions(self, val: int):
        self.max_num_actions = val

    def set_sequence_list(self, val: list):
        self.seq_list = val

    #######################################################
    #           Methods for UE environment setup
    #######################################################

    def set_environment(self):
        '''
        Method that will set the UE environment to a default state.
        Specifically
        1. setting the camera to the first poi's first location
        :return:
        '''
        unreal.log("::set_environment()")

        # Set the camera to track the moving point light
        track_set = unreal.CameraLookatTrackingSettings(
                                enable_look_at_tracking=True)
        self.camera_obj.set_editor_property("lookat_tracking_settings",
                                            track_set)
        self.set_camera_tracking(self.cc_tracker_obj)

        ###
         # Set initial camera location
        self.set_obj_loc(poi_obj=self.cc_tracker_obj,
                         new_loc=self.plate_corner_refs[0],
                         abs_loc_flag=True)
        self.set_obj_loc(poi_obj=self.camera_obj,
                         new_loc=self.cam_home,
                         abs_loc_flag=False)
        self.set_obj_loc(poi_obj=self.cam_light_obj,
                         new_loc=self.cam_light_home,
                         abs_loc_flag=False)

        ###
        # Reset class counts
        self.frame_count = 0
        self.action_count = 0

        # Reset the state machine
        self.debug_current_plate = -1
        self.debug_current_basecolor = -1
        self.debug_current_map_set = -1
        self.debug_current_plate_offset = -1
        self.debug_current_cam_angle = -1
        self.debug_current_lighting = -1


    #######################################################
    #           Test Sequences
    #######################################################

    def start(self, seq: int):
        self.editor.clear_actor_selection_set()
        self.editor.set_selected_level_actors([self.camera_obj])

        # TODO - implement checks to make sure all needed variables are set
        #  like maximum_frame_count and frame_buffer, and max_num_actions

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
        # Potentially end the thread
        terminate = False
        if self.frame_count >= self.max_frame_count:
            unreal.log_error(f"<<<<< Hit max_frame_count {self.max_frame_count}. "
                             f"Terminating <<<<<<")
            terminate = True

        elif self.action_count >= self.max_num_actions:
            unreal.log_error(f"<<<<< Hit max_num_actions. {self.max_num_actions}. "
                             "Terminating <<<<<<")
            terminate = True

        #######################################################
        #           Perform Actions at ticks
        #######################################################

        # 1) Reset at end of actions or as a protective measure
        if terminate:
            self.set_environment()

            self.output_image_csv(path=f"{self.output_path}/{self.output_dir}",
                                  filename="synth-welds.csv")

            # Reset before the end of the thread
            self.action_count += 1
            unreal.log("Unregister callback")
            unreal.unregister_slate_post_tick_callback(
                self.slate_post_tick_handle)

            self.in_progress = False

        # 2) Perform action sequence
        elif self.action_count < self.max_num_actions:

            # Only execute actions every 'frame_buffer' frames
            if self.frame_count % self.frame_buffer == 0:
                try:
                    terminate = self.run_image_seq1(self.action_count)
                except Exception:
                    unreal.log_error("CAUGHT ERROR: "
                             f"plate {self.debug_current_plate}"
                             f"basecolor {self.debug_current_basecolor}"
                             f"map_set {self.debug_current_map_set}, "
                             f"plate_offset {self.debug_current_plate_offset}, "
                             f"angle {self.debug_current_cam_angle}, "
                             f"lighting {self.debug_current_lighting}"
                             f"\n\nState: {self.print_thread_state()}")
                    unreal.log("Unregister callback")
                    unreal.unregister_slate_post_tick_callback(
                        self.slate_post_tick_handle)
                    terminate = True

                if terminate:
                    self.action_count = self.max_num_actions

        self.frame_count += 1

    def run_image_seq1(self, action_num):
        print(f"---- Performing Action {action_num}/{self.max_num_actions}")
        print(f"*** DEBUG: debug #s: {self.debug_current_plate}/"
              f"{self.debug_current_basecolor}/{self.debug_current_map_set}/"
              f"{self.debug_current_plate_offset}/"
              f"{self.debug_current_cam_angle}/{self.debug_current_lighting}/"
              )
        error = False

        # Value of state machine
        sum_of_states = self.debug_current_plate + \
                        self.debug_current_basecolor + \
                        self.debug_current_map_set + \
                        self.debug_current_plate_offset + \
                        self.debug_current_cam_angle


        if sum_of_states <= 0:
            # Initial setup - no screenshot
            error = error or self.set_scene(
                                    plates_id=0, basecolor_id=0,
                                    map_set_id=0, plate_offset_id=0,
                                    angle_id=0)

        if 0 <= sum_of_states < self.max_num_actions:
            # Execute image pipeline - screenshot for each setting configuration
            set = self.seq_list[action_num]
            error = error or self.set_scene(
                                    plates_id=set[0], basecolor_id=set[1],
                                    map_set_id=set[2], plate_offset_id=set[3],
                                    angle_id=set[4])

            # Setup the resulting image & log in the .csv record
            image_name, label = self.create_image_name(
                                    action_num=action_num,
                                    plate_num=self.debug_current_plate,
                                    color_num=self.debug_current_basecolor,
                                    map_set_num=self.debug_current_map_set,
                                    offset_num=self.debug_current_plate_offset,
                                    angle_num=self.debug_current_cam_angle)
            abs_filepath = f"{self.output_path}\\{self.output_dir}\\" \
                           f"images\\{image_name}"
            self.log_image(abs_filepath=abs_filepath, label=label)

            # Take a screenshot
            self.take_HighResShot2(abs_filepath=abs_filepath)

            self.action_count += 1

        if sum_of_states >= self.max_num_actions:
            unreal.log_warn("*** WARN: Reached max_num_actions "
                            f"{self.max_num_actions}")
            error = error or self.set_scene(
                                        plates_id=0, basecolor_id=0,
                                        map_set_id=0, plate_offset_id=0,
                                        angle_id=0)


        return error

    def set_scene(self, plates_id: int, basecolor_id: int, map_set_id: int,
                  plate_offset_id: int, angle_id: int):
        terminate = False

        try:
            if plates_id != self.debug_current_plate:
                print("change plate")

                # move cc_tracker and camera to the new plate corner
                self.set_obj_loc(poi_obj=self.cc_tracker_obj,
                                 new_loc=self.plate_corner_refs[
                                                plates_id],
                                 abs_loc_flag=True
                                 )

                self.debug_current_plate = plates_id

            if basecolor_id != self.debug_current_basecolor:
                print("change basecolor")

                # Set map set based on current debug counts and increment
                # internal counts.
                terminate = self.set_param(mat_path=self.mat_path,
                                           param_name=self.mat_color_sw,
                                           val=basecolor_id)

                self.debug_current_basecolor = basecolor_id

            if map_set_id != self.debug_current_map_set:
                print("change map_set")

                # Set map set based on current debug counts and increment
                # internal counts.
                for key in self.mat_map_set_sws.keys():
                    terminate = self.set_param(
                                        mat_path=self.mat_path,
                                        param_name=self.mat_map_set_sws[key],
                                        val=map_set_id)

                self.debug_current_map_set = map_set_id

            if plate_offset_id != self.debug_current_plate_offset:
                print("change plate_offset")

                corner_ref = self.plate_corner_refs[self.debug_current_plate]
                offset = self.offsets_plate[plate_offset_id]
                new_loc = [round(corner_ref[i] + offset[i], 2) for i in
                           range(3)]

                self.set_obj_loc(poi_obj=self.cc_tracker_obj,
                                 new_loc=new_loc,
                                 abs_loc_flag=True)

                self.debug_current_plate_offset = plate_offset_id

            if angle_id != self.debug_current_cam_angle or \
                angle_id != self.debug_current_lighting:
                print("change cam angle & lighting")

                # Move the Camera
                offset = self.offsets_cam[self.action_count]
                new_loc = [round(self.cam_home[i] + offset[i], 2) for i in
                           range(3)]

                self.set_obj_loc(poi_obj=self.camera_obj,
                                 new_loc=new_loc,
                                 abs_loc_flag=False)

                # Move the accompanying light
                offset = self.offsets_light[self.action_count]
                new_loc = [round(self.cam_light_home[i] + offset[i], 2) for \
                                                                i in range(3)]

                self.set_obj_loc(poi_obj=self.cam_light_obj,
                                 new_loc=new_loc,
                                 abs_loc_flag=False)

                self.debug_current_cam_angle = angle_id
                self.debug_current_lighting = angle_id

        except Exception:
            unreal.log_error("ERROR - ::set_scene(): "
                             "Caught issue with sequence number "
                             f"{plates_id}/{basecolor_id}/{map_set_id}/"
                             f"{plate_offset_id}/{angle_id}/")
            terminate = True

        return terminate


    def print_thread_state(self):
        msg = f"set_camera_to_corner: {self._state['set_camera_to_corner']}, " \
                f"set_basecolor: {self._state['set_basecolor']}, " \
                f"set_map_set: {self._state['set_map_set']}, " \
                f"set_plate_offset: {self._state['set_plate_offset']}, " \
                f"get_snapshots: {self._state['get_snapshots']}, " \
                f"shift_to_new_model: {self._state['shift_to_new_model']}, " \
                f"end_of_models_terminate_early: {self._state['end_of_models_terminate_early']}"

        return msg

    #######################################################
    #           CineCameraActor Methods
    #######################################################

    def take_HighResShot2(self, abs_filepath):
        cmd = f"HighResShot 2 filename=\"{abs_filepath}\""
        unreal.SystemLibrary.execute_console_command(
            self.editor.get_editor_world(), cmd)

    def set_camera_tracking(self, poi_obj):
        track_set = unreal.CameraLookatTrackingSettings(
            enable_look_at_tracking=True,
            actor_to_track=poi_obj,
            relative_offset=unreal.Vector(z=0))
        self.camera_obj.set_editor_property("lookat_tracking_settings", track_set)

    def seq_step_plate_basecolor(self):

        seq_pos = self.debug_current_basecolor

        try:
            unreal.log(f"*** basecolor #{seq_pos}")

            # TODO - Flip material switches as needed

            self.debug_current_basecolor = seq_pos + 1
            force_terminate = False

        except IndexError:
            unreal.log_warning(f"*** ERROR: Attempting too many basecolors "
                               f" ({self.debug_current_basecolor}). "
                               f"Only {self.num_basecolors} num_basecolors "
                               f"are specified. Self terminating")
            force_terminate = True

        return force_terminate


    def seq_step_map_set(self):

        seq_pos = self.debug_current_map_set

        try:
            unreal.log(f"*** map_set #{seq_pos}")

            # TODO - Flip material switches as needed

            self.debug_current_map_set = seq_pos + 1
            force_terminate = False

        except IndexError:
            unreal.log_warning(f"*** ERROR: Attempting too many map_sets "
                               f" ({self.debug_current_map_set}). "
                               f"Only {self.num_map_sets} map_sets "
                               f"are specified. Self terminating")
            force_terminate = True

        return force_terminate

    def seq_step_plate_offset(self):

        seq_pos = self.debug_current_plate_offset

        try:
            unreal.log(f"*** offset_plate #{seq_pos}")
            corner_ref = self.plate_corner_refs[self.debug_current_plate]
            offset = self.offsets_plate[self.debug_current_plate_offset]
            new_loc = [round(corner_ref[i]+offset[i],2) for i in range(3)]

            self.set_obj_loc(poi_obj=self.cc_tracker_obj,
                             new_loc=new_loc,
                             abs_loc_flag=True)

            self.debug_current_plate_offset = seq_pos + 1
            force_terminate = False

        except IndexError:
            unreal.log_warning(f"*** ERROR: Attempting too many camera "
                               f"offsets_cam"
                               f" ({self.debug_current_plate_offset}). "
                               f"Only {len(self.offsets_plate)} plate offsets "
                               f"are specified. Self terminating")
            force_terminate = True

        return force_terminate

    def seq_step_cam_angle_offset(self):

        plate_offset_pos = self.debug_current_plate_offset
        seq_pos = self.debug_current_cam_angle

        try:
            unreal.log(f"*** offset_cam loc #{seq_pos}")
            offset = self.offsets_cam[plate_offset_pos][seq_pos]
            new_loc = [round(self.cam_home[i]+offset[i],2) for i in range(3)]

            self.set_obj_loc(poi_obj=self.camera_obj,
                             new_loc=new_loc,
                             abs_loc_flag=False)
            self.debug_current_cam_angle = seq_pos + 1
            force_terminate = False

        except IndexError:
            unreal.log_warning(f"*** ERROR: Attempting too many camera "
                               f"offsets_cam"
                               f" ({self.debug_current_cam_angle}). Only "
                               f"{len(self.offsets_cam[plate_offset_pos])} "
                               f"camera angle offsets are specified. "
                               f"Self terminating")
            force_terminate = True

        return force_terminate

    def seq_step_light_offset(self):

        plate_offset_pos = self.debug_current_plate_offset
        seq_pos = self.debug_current_lighting

        try:
            unreal.log(f"*** offset_light loc #{seq_pos}")
            offset = self.offsets_light[plate_offset_pos][seq_pos]
            new_loc = [round(self.cam_home[i]+offset[i],2) for i in range(3)]

            self.set_obj_loc(poi_obj=self.cam_light_obj,
                             new_loc=new_loc,
                             abs_loc_flag=False)
            self.debug_current_lighting = seq_pos + 1
            force_terminate = False

        except IndexError:
            unreal.log_warning(f"*** ERROR: Attempting too many light "
                               f"offsets_light"
                               f" ({self.debug_current_lighting}). Only "
                               f"{len(self.offsets_light[plate_offset_pos])} "
                               f"light offsets are specified. "
                               f"Self terminating")
            force_terminate = True

        return force_terminate

    #######################################################
    #           Object Actor Methods
    #######################################################

    def set_obj_loc(self, poi_obj, new_loc, abs_loc_flag: int):

        dest_vect = unreal.Vector(
            x=new_loc[0],
            y=new_loc[1],
            z=new_loc[2])

        if hasattr(poi_obj, 'root_component'):
            try:
                poi_obj.root_component.set_editor_property("absolute_location",
                                                           abs_loc_flag)
                poi_obj.root_component.set_editor_property("relative_location",
                                                            dest_vect)
            except:
                unreal.log_error("::set_obj_loc() - caught issue with setting "
                           "abs_loc or rel_loc")
        else:
            unreal.log_error("*** ERROR - set_obj_loc() - Cannot set location "
                             "for the desired poi_obj. No 'root_component' "
                             "property to access to move the obj.")

    #######################################################
    #           Object Material Methods
    #######################################################

    def set_param(self, mat_path, param_name: str, val: float):
        terminate = False

        try:
            obj = unreal.load_object(None, mat_path)
            assert obj
        except:
            unreal.log_error("*** ERROR: could not load Unreal Object at path "
                             f"{mat_path}")
            terminate = True

        try:
            # Set material parameter using MaterialEditingLibrary() methods
            editor = unreal.MaterialEditingLibrary()

            editor.set_material_instance_scalar_parameter_value(obj,
                                                                param_name,
                                                                val)
        except:
            unreal.log_error("*** ERROR: set unreal material instance param "
                             f"{param_name} to value {val}")
            terminate = True

        return terminate

    #######################################################
    #           log images
    #######################################################

    def create_output_dir(self, path):
        now = datetime.now()
        start = now.strftime("%Y_%m_%d-%H_%M_%S")
        # BASE_PATH = os.path.join(path, f"{start}-pipeline")
        new_dir = start+"-pipeline"

        try:
            os.mkdir(os.path.join(path, new_dir))
            self.output_dir = new_dir
        except OSError as error:
            print(error)

    def create_image_name(self, action_num: int, plate_num: int,
                            color_num: int, map_set_num: int, offset_num: int,
                            angle_num: int):
        try:
            label = "Defect" if self.debug_current_map_set in \
                                self.defect_map_sets else "Good"
            debug_msg = f"{plate_num}_{color_num}_{map_set_num}_" \
                        f"{offset_num}_{angle_num}"
        except Exception:
            unreal.log_error("*** ERROR: issue in ::create_image_name().")

        image_name = f"{label}_{action_num}-{debug_msg}.png"

        return image_name, label

    def log_image(self, abs_filepath, label):
        #self.log_df.loc[len(self.log_df.index)] = [abs_filepath, label]
        self.log_df = self.log_df + f"{abs_filepath},{label}\n"

    def output_image_csv(self, path, filename):
        print("Run ::output_image_csv()")

        try:
            #self.log_df.to_csv(path)

            f = open(f"{path}/{filename}", "w")
            f.write(self.log_df)
            f.close()
            unreal.log_warning(f"Write csv file to {path}")

        except NameError:
            print(f"**** WARN: Dataframes to {path} was not defined")


#################################################
#               Helper Methods
#################################################


def link_ue_objects():
    plates = []
    cylinders = []
    for y in actors:
        name = y.get_name()

        if name == "Main_Camera":
            # Ensure that the camera Actor is selected for screenshots
            cam = y
            print("*** DEBUG: Found Camera actor")

        if 'CineCamera_Tracker' == name:
            cc_tracker = y
            print("*** DEBUG: Found CineCamera_Tracker PointLight of the shot")

        if 'plate'.lower() in name.lower():
            plates.append(y)
            print(f"*** DEBUG: Found {name} object")

        if 'cylinder' in name.lower():
            cylinders.append(y)
            print(f"*** DEBUG: Found {name} object")

        if 'pointlight_camera' in name.lower():
            cam_light = y
            print("*** DEBUG: Found the Camera's PointLight actor")

    print(f"plates = {len(plates)}\tcylinder = {len(cylinders)}")

    try:
        actor_dic = {"level_editor": unreal.EditorLevelLibrary(),
                     "camera": cam,
                     "cc_tracker": cc_tracker,
                     "points_of_interest_obj": plates + cylinders,
                     "pointlight_camera": cam_light
                     }

    except:
        raise("*** ERROR - ::link_ue_objects() - Could not find all desired "
              "UE actors. Terminating")

    return actor_dic


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


def define_weld_pts(pixel_spacing: int,
                    dist_from_border: int,
                    cam_height_from_plate: int):
    '''
    Define the x,y coordinates for all weld locations under review


    :param pixel_spacing:
    :return: list of [x,y] entries specifying offsets to take screenshots of.
    '''

    # Locations for horizontal welds
    y_axis_intersects = [1365, 2732]
    x_locs = range(0+dist_from_border, 4096-dist_from_border, pixel_spacing)

    pts_01 = []
    for weld_row_y in y_axis_intersects:
        for x in x_locs:
            pts_01.append([x, weld_row_y, cam_height_from_plate, 'horizontal'])

    # Locations for vertical welds
    x_axis_intersects = [[308, 1333, 2358, 3382],  # Intersection of x-axis
                          [141, 1169, 2193, 3218],
                          [651, 1675, 2699, 3723]]

    pts_02 = []
    for weld_col_x in x_axis_intersects[0]:
        y_locs = range(0+dist_from_border, y_axis_intersects[0], pixel_spacing)
        for y in y_locs:
            pts_02.append([weld_col_x, y, cam_height_from_plate, 'vertical'])

    for weld_col_x in x_axis_intersects[1]:
        y_locs = range(y_axis_intersects[0], y_axis_intersects[1],
                       pixel_spacing)
        for y in y_locs:
            pts_02.append([weld_col_x, y, cam_height_from_plate, 'vertical'])

    for weld_col_x in x_axis_intersects[2]:
        y_locs = range(y_axis_intersects[1], 4096-dist_from_border, pixel_spacing)
        for y in y_locs:
            pts_02.append([weld_col_x, y, cam_height_from_plate, 'vertical'])

    return pts_01 + pts_02

def correct_weld_pts_for_ue(plate_offsets: list,
                            bmp_x: int, bmp_y: int,
                            plate_x: int, plate_y: int, cam_height: int):
    for i in range(len(plate_offsets)):
        # only take the x,y coordinate of the offset
        entry = plate_offsets[i][:2]

        # Normalize the resulting x & y based on the bitmap size 4096x4096
        entry[0] = entry[0]/bmp_x * plate_x
        entry[1] = entry[1] / bmp_x * plate_y

        # add a z-axis location for the camera to make a 3-d vector
        entry.append(cam_height)

        plate_offsets[i] = entry

    return plate_offsets

def set_random_offsets(num_of_pts: int,
                        x_tol: int, y_tol: int, z: int, z_tol: list):
    obj_offsets = []

    random.seed(None)

    k = 0
    for i in range(num_of_pts):
        offset_x = random.randint(-x_tol, x_tol)
        offset_y = random.randint(-y_tol, y_tol)
        offset_z = z + random.randint(z_tol[0], z_tol[1])
        obj_offsets.append([offset_x, offset_y, offset_z])

    print(obj_offsets[:5])

    return obj_offsets


def define_sequence_ints(num_plates: list, basecolor_ids: list,
                         map_set_ids: list, num_offsets_per_plate: int,
                         num_angles_per_ofset: int):
    seq_list = []

    for a in range(num_plates):
        for b in basecolor_ids:
            for c in map_set_ids:
                for d in range(num_offsets_per_plate):
                    for e in range(num_angles_per_ofset):
                        seq_list.append([a, b, c, d, e])

    return seq_list


if __name__=="__main__":

    #################################################
    #               User Settings
    #################################################

    image_output_path = "..\\_data\\welding\\synth"


    ###
    # Load the needed Material Instance and all used switches
    MATERIAL_PATHS = "/Game/ImagePipeline-Welds/" \
                     "M_MainPlate_Inst.M_MainPlate_Inst"

    MATERIAL_PARAMS = {'basecolor': "sw_basecolor",
                       'AmbientOcclusion': "sw_AmbientOcclusion",
                       'Metallic': "sw_Metallic",
                       'Normal': "sw_Normal",
                       'Roughness': "sw_Roughness"}

    basecolor_ids = list(range(6))
    map_set_ids = list(range(5))
    angles_per_offset = 4


    ###
    # Set geographic locations for the screenshots to follow.

    cam_height_from_plate = 6
    cc_tracker_light_height_from_plate = 200

    pixel_spacing_bw_locs = 400     # If changed... must recreate map_sets
    dist_from_border = 125          # If changed... must recreate map_sets
    cc_tracker_abs_height = 35      # Do not change
    plate_offsets = define_weld_pts(pixel_spacing=pixel_spacing_bw_locs,
                                    dist_from_border=dist_from_border,
                                    cam_height_from_plate=cc_tracker_abs_height)
    plate_offsets = correct_weld_pts_for_ue(plate_offsets=plate_offsets,
                                            bmp_x=4096, bmp_y=4096,
                                            plate_x=800, plate_y=800,
                                            cam_height=cam_height_from_plate)

    # DEBUG override
    #plate_offsets = plate_offsets[:2]

    plate_ids_ref_left_corners = [[-2150, -1300, 35]]

    cam_home = [[0, 0, cam_height_from_plate],
                [0, 0, cc_tracker_light_height_from_plate]]

    FRAME_BUFFER = 30

    #################################################
    #               Preparation
    #################################################

    # Code to print out all or specific actors in a loaded level
    actors = unreal.EditorLevelLibrary().get_all_level_actors()
    actor_dic = link_ue_objects()

    # Calculate the number of actions that the test will execute and the
    # projected maximum number of frames to complete the task
    num_plates = len(plate_ids_ref_left_corners)
    num_basecolors = len(basecolor_ids)
    num_map_sets = len(map_set_ids)
    num_offsets_per_plate = len(plate_offsets)

    max_num_actions = num_plates * num_basecolors * num_map_sets * \
                  num_offsets_per_plate * angles_per_offset
    print(f"*** DEBUG: {max_num_actions} expected.\n")
    num_req_frames = max_num_actions * FRAME_BUFFER * 5.0

    seq_list = define_sequence_ints(num_plates=num_plates,
                                 basecolor_ids=basecolor_ids,
                                 map_set_ids=map_set_ids,
                                 num_offsets_per_plate=num_offsets_per_plate,
                                 num_angles_per_ofset=angles_per_offset)


    ###
    # Add randomness to the camera and accompanying light

    cam_offsets_per_plate_offset = set_random_offsets(
                                        num_of_pts=max_num_actions,
                                        x_tol=15,
                                        y_tol=15,
                                        z=cam_height_from_plate,
                                        z_tol=[0, 15]
                                    )
    light_offsets_per_plate_offset = set_random_offsets(
                                        num_of_pts=max_num_actions,
                                        x_tol=150,
                                        y_tol=150,
                                        z=cc_tracker_light_height_from_plate,
                                        z_tol=[-50, 100]
                                    )

    #################################################
    #               class initialization
    #################################################

    # Setup the custom UE Execution object to collect all screenshots
    test1 = UE_ImagePipeline(ue_objs=actor_dic,
                            reference_corners=plate_ids_ref_left_corners,
                            num_basecolors=num_basecolors,
                            num_map_sets=num_map_sets,
                            plate_offsets=plate_offsets,
                            cam_offsets_per=cam_offsets_per_plate_offset,
                            light_offsets_per=light_offsets_per_plate_offset,
                            cam_home=cam_home,
                            material_path=MATERIAL_PATHS,
                            material_params_dict=MATERIAL_PARAMS,
                            output_path_of_run=image_output_path)

    # test1.calc_cam_cartesian_locs(range_rho=camera_angles['rho'],
    #                               range_alpha=camera_angles['alpha'],
    #                               range_phi=camera_angles['phi'])

    test1.set_max_frames(num_req_frames)
    test1.set_frame_buffer(FRAME_BUFFER)
    test1.set_max_num_actions(max_num_actions)
    test1.set_sequence_list(seq_list)

    #################################################
    #               Start of Execution
    #################################################

    unreal.log_warning("----- START TEST ------")
    test1.set_environment()
    test1.start(1)
    unreal.log_warning("*** DEBUG: PRESET COMPLETE!!!")