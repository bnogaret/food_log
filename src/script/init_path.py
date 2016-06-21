"""
Set up path to thesis_lib for the different script files.
"""

import os
import sys


def add_path_to_sys(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)

# Add thesis_lib to PYTHONPATH
lib_path = os.path.join(this_dir, '..')
add_path_to_sys(lib_path)
