import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from thesis_lib.bbox import get_overlap_ratio_bbox, get_correct_bbox

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
    
    def test_get_correct_bbox_simple(self):
        gt = np.asarray([[0, 0, 50, 50], [100, 100, 200, 200], [0, 0, 1000, 1000], [500, 500, 600, 600]])
        predicted = np.asarray([[0, 0, 40, 40], [300, 300, 400, 400], [100, 100, 900, 900]])
        
        res_correct, res_gt, res_metrics = get_correct_bbox(gt, predicted)
        
        expect_correct = np.asarray([[0, 0, 40, 40], [100, 100, 900, 900]])
        expect_gt = np.asarray([[0, 0, 50, 50], [0, 0, 1000, 1000]])
        expec_metrics = np.asarray((2/5, 2/3, 2/4))
        
        assert_array_equal(res_correct, expect_correct)
        assert_array_equal(res_gt, expect_gt)
        assert_array_almost_equal(res_metrics, expec_metrics)
    
    def test_get_correct_bbox_multiple_predicted_bbox_for_one_gt_bbox(self):
        # Test multiple bbox for one gt
        gt = np.asarray([[0, 0, 50, 50], [100, 100, 200, 200]])
        predicted = np.asarray([[0, 0, 40, 40], [10, 10, 50, 50], [0, 0, 50, 50]])
        
        res_correct, res_gt, res_metrics = get_correct_bbox(gt, predicted)
        
        expect_correct = np.asarray([[0, 0, 40, 40]])
        expect_gt = np.asarray([[0, 0, 50, 50]])
        expec_metrics = np.asarray((1/4, 1/3, 1/2))
        
        assert_array_equal(res_correct, expect_correct)
        assert_array_equal(res_gt, expect_gt)
        assert_array_almost_equal(res_metrics, expec_metrics)
    
    def test_get_correct_with_category_column(self):
        # Check if i can pass with a fifth column for the category of bbox
        gt = np.asarray([[0, 0, 50, 50, 1], [0, 0, 1000, 1000, 2]])
        predicted = np.asarray([[0, 0, 40, 40], [100, 100, 900, 900]])
        
        res_correct, res_gt, res_metrics = get_correct_bbox(gt, predicted)
        
        expect_correct = np.asarray([[0, 0, 40, 40], [100, 100, 900, 900]])
        expect_gt = np.asarray([[0, 0, 50, 50, 1], [0, 0, 1000, 1000, 2]])
        expec_metrics = np.asarray((1, 1, 1))
        
        assert_array_equal(res_correct, expect_correct)
        assert_array_equal(res_gt, expect_gt)
        assert_array_almost_equal(res_metrics, expec_metrics)

