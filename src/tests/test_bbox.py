import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from thesis_lib.bbox import get_overlap_ratio_bbox, get_accuracy_bbox

# python -m unittest discover


class OverlapTest(unittest.TestCase):
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
    
    def test_get_accuracy_bbox_simple(self):
        gt = np.asarray([[0, 0, 50, 50], [100, 100, 200, 200], [0, 0, 1000, 1000], [500, 500, 600, 600]])
        predicted = np.asarray([[0, 0, 40, 40], [300, 300, 400, 400], [100, 100, 900, 900]])
        
        res_array, res_metrics = get_accuracy_bbox(gt, predicted)
        expec_metrics = np.asarray((2/5, 2/3, 2/4))
        expec_array = np.asarray([[0, 0, 40, 40], [100, 100, 900, 900]])
        
        assert_array_almost_equal(res_metrics, expec_metrics)
        assert_array_equal(res_array, expec_array)
    
    def test_get_accuracy_bbox_multiple_predicted_bbox_for_one_gt_bbox(self):
        # Test multiple bbox for one gt
        gt = np.asarray([[0, 0, 50, 50], [100, 100, 200, 200]])
        predicted = np.asarray([[0, 0, 40, 40], [10, 10, 50, 50], [0, 0, 50, 50]])
        
        res_array, res_metrics = get_accuracy_bbox(gt, predicted)
        expec_metrics = np.asarray((1/4, 1/3, 1/2))
        expec_array = np.asarray([[0, 0, 40, 40]])
        
        assert_array_almost_equal(res_metrics, expec_metrics)
        assert_array_equal(res_array, expec_array)

