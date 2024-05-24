'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import unreal


# noinspection PyUnresolvedReferences
class MyClass(object):
    def __init__(self, frames_buffer: int) -> None:
        self.frame_count = 0
        self.max_count = 1000
        self.frame_buffer = frames_buffer

    def start(self) -> None:
        self.slate_post_tick_handle = unreal.register_slate_post_tick_callback(self.tick)
        self.frame_count = 0

    def tick(self, delta_time: float) -> None:
        print(self.frame_count)
        self.frame_count += 1
        if self.frame_count >= self.max_count:
            unreal.unregister_slate_post_tick_callback(
                self.slate_post_tick_handle)
        elif self.frame_count % self.frame_buffer == 0:
            print("Do things for one callback")


frames_bw_ticks = 10
test = MyClass(frames_bw_ticks)
test.start()

