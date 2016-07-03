"""
Contains functions for that are only applicable for the UEFCFOOD 256 and UECFOOD 100 dataset.
"""

from .parser import get_name_and_category, read_bb_info_txt

from .utils_uecfood import get_list_image_file


__all__ = ['get_name_and_category',
           'read_bb_info_txt',
           'get_list_image_file']


