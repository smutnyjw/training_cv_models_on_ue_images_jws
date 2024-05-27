'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''


import time
import unreal

unreal.SystemLibrary.execute_console_command(unreal.EditorLevelLibrary.get_editor_world(), "HighResShot 2")

# unreal.EditorLoadingAndSavingUtils.flush_async_loading()
# editor_util = unreal.AnalyticsLibrary()
# editor_util.flush_events()
# unreal.AsyncDelayComplete()
# unreal.EditorTick().tick(unreal.EditorLevelLibrary, 0.0)

# editor = unreal.EditorLoadingAndSavingUtils
# editor.save_dirty_packages(True, True)

unreal.SystemLibrary.execute_console_command(unreal.EditorLevelLibrary.get_editor_world(), "HighResShot 2")