'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''

import unreal
import os

editor = unreal.EditorLevelLibrary()
# abs_filepath = "C:\\Users\\johns\\PycharmProjects\\VT\\MENG_Project\\test_02" \
#                ".png"
abs_filepath = os.path.join("C:\\", "Users", "johns",
                                "PycharmProjects", "VT",
                                "MENG_Project", "test_02.png")
cmd = f"HighResShot 2 filename=\"{abs_filepath}\""
unreal.SystemLibrary.execute_console_command(editor.get_editor_world(), cmd)