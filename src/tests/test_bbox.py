import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import unittest
import numpy as np

from thesis_lib.bbox import get_overlap_ratio_bbox

# python -m unittest discover


class AccuracyTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_simple_overlap(self):
        bbox = np.asarray([[0, 0, 100, 100], [50, 50, 100, 100]])
        res = get_overlap_ratio_bbox(bbox[0], bbox[1])
        
        self.assertEqual(res, .25)
    
    def test_non_overlap(self):
        bbox = np.asarray([[0, 0, 50, 50], [100, 100, 200, 200]])
        res = get_overlap_ratio_bbox(bbox[0], bbox[1])
        
        self.assertEqual(res, 0.0)
