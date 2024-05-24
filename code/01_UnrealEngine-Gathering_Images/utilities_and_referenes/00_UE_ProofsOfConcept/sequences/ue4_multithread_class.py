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

class MoveCamera(object):
    def __init__(self, iterations):
        # Threading setup
        self.frame_count = 0
        self.ticks_per_action = 5
        self.max_count = iterations*self.ticks_per_action

        # Camera setup
        self.cam = None
        self.cam_scene = None
        self.cam_home = unreal.Vector(x=0, y=0, z=0)
        self.get_cam_info()


    def get_cam_info(self):
        self.cam = unreal.EditorLevelLibrary.get_selected_level_actors()
        self.cam_scene = self.cam[0].scene_component
        self.cam_home = unreal.Vector(x=-150, y=-60, z=150)
        print(f"HOME Camera location: {self.cam_home}")

    def start(self):
        if self.frame_count != 0:
            print("Please wait until first call is done.")
        else:
            self.slate_post_tick_handle = \
                unreal.register_slate_post_tick_callback(self.tick)
            self.frame_count = 0

            print("Test")

    def tick(self, delta_time: float):
        print(self.frame_count)

        # Potentially end the thread
        if self.frame_count > self.max_count:
            # End tick and any changes
            unreal.unregister_slate_post_tick_callback(
                self.slate_post_tick_handle)


        #######################################################
        #           Perform Actions at ticks
        #######################################################

        # 1) Reset at end of actions
        if self.frame_count == self.max_count:
            # Reset before the end of the thread
            self.teleport_camera_home(home=self.cam_home)

        # 2) Move camera then take a snapshot
        elif self.frame_count % self.ticks_per_action == 0:
            self.take_HighResShot2()
            self.teleport_camera(_x=self.cam_home.x + 0*self.frame_count,
                                 _y=self.cam_home.y + 0*self.frame_count,
                                 _z=self.cam_home.z + 20*self.frame_count)

        self.frame_count += 1

    #######################################################
    #           CineCameraActor Methods
    #######################################################

    def take_HighResShot2(self):
        print("***** Take screenshot of current Editor")
        unreal.SystemLibrary.execute_console_command(
            unreal.EditorLevelLibrary.get_editor_world(), "HighResShot 2")


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


test1 = MoveCamera(3)
test1.start()




