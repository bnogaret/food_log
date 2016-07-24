import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import unittest
import numpy as np

import constants as const
from thesis_lib.uecfood import get_ground_truth_bbox

# python -m unittest discover


class GetGroundTruthBboxTest(unittest.TestCase):
    def setUp(self):
        self.df = get_ground_truth_bbox(const.PATH_TO_ROOT_UECFOOD256, verbose=False)

    def test_simple_overlap(self):
        # Check right number of bbox
        self.assertEqual(self.df.shape, (31645, 8))
        # Number of different files
        self.assertEqual(len(self.df._abs_path.unique()), 31395)
        # Check right number of different category
        self.assertEqual(len(self.df._cat.unique()), 256)
        # Check all the values are assigned
        self.assertFalse(self.df.isnull().values.any())
        
    def test_multi_item(self):
        # Check the right behaviour for some images
        self.assertEqual(self.df.loc[(self.df._img_name == 4004) & (self.df._multi_item == True)].shape, (4, 8))
        
        self.assertEqual(self.df.loc[(self.df._img_name == 14930) & (self.df._multi_item == True)].shape, (6, 8))
        
        self.assertEqual(self.df.loc[(self.df._img_name == 97) & (self.df._multi_item == True)].shape, (3, 8))
        self.assertEqual(self.df.loc[(self.df._img_name == 97) & (self.df._multi_item == False)].shape, (1, 8))
        
        self.assertEqual(self.df.loc[(self.df._img_name == 199) & (self.df._multi_item == True)].shape, (0, 8))
        self.assertEqual(self.df.loc[(self.df._img_name == 199) & (self.df._multi_item == False)].shape, (2, 8))
        
        self.assertEqual(self.df.loc[(self.df._img_name == 335214) & (self.df._multi_item == False)].shape, (1, 8))
        self.assertEqual(self.df.loc[(self.df._img_name == 335214) & (self.df._multi_item == True)].shape, (0, 8))

