"""
File including global constants.

.. warning::
    Change the values according to your project.

Examples
--------
>>> import constants as const
>>> print(const.PATH_TO_UEFCFOOD256)
/home/nogaret/.virtualenvs/thesis/project/src/../data/UECFOOD256/
"""

import os


__all__ = ['PATH_TO_UECFOOD256']


PATH_CURRENT_DIRECTORY = os.path.dirname(__file__)
PATH_TO_UECFOOD256 = PATH_CURRENT_DIRECTORY + "/../data/UECFOOD256/"
"""
str: Path to the root directory including the UEC FOOD 256 dataset.

http://foodcam.mobi/dataset256.html
"""
