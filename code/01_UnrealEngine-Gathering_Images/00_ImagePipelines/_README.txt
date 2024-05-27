/01_UnrealEngine-Gathering_Images

This directory contains two sets of python scripts that each individually
interact with an open and running Unreal Engine Editor window. Each .py file
operates independently, they do not reference each other.

/00_ImagePipelines contains the .py script that generate the CatDog and Welding
synthetic datasets (assuming that you have setup an UE Environment with all of
the required actor objects and material instances (with material parameters). In
addition, the 'class.py' file is the base class for a user to execute their
own 'callback' based script to be executed every x frames of an UE Editor
environment.

/utilities_and_references contains a variety of scripts based on 'class.py' and
single use scripts meant to show the user how to interact with their UE Editor
from the UE Python API. Functions include: moving actors, changing material
parameters, and moving the camera CineCamera actor.
