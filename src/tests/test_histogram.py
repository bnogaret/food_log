import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import unittest

import numpy as np
from numpy.testing import assert_allclose

import constants as const
from thesis_lib.histogram import color_histogram


# https://github.com/scikit-image/scikit-image/blob/401c1fd9c7db4b50ae9c4e0a9f4fd7ef1262ea3c/skimage/io/tests/test_histograms.py


class ColorHistogramTest(unittest.TestCase):
    def test_marginal_count(self):
        img = np.ones((50, 50, 3), dtype=np.uint8)
        
        hist = color_histogram(img, bins=5, ranges=(0, 1), normalization=False)
        
        self.assertEqual(50 * 50 * 3, np.sum(hist))
    
    def test_marginal_basic(self):
        channel = np.arange(255).reshape(51, 5)
        img = np.empty((51, 5, 3), dtype=np.uint8)
        img[:, :, 0] = channel
        img[:, :, 1] = channel
        img[:, :, 2] = channel
        
        hist = color_histogram(img, bins=255, ranges=(0, 255))

        hist = hist.reshape(255, 3)
        
        assert_allclose(hist[:, 0], np.ones(255)/255, rtol=1e-04, atol=1e-05)
        assert_allclose(hist[:, 1], np.ones(255)/255, rtol=1e-04, atol=1e-05)
        assert_allclose(hist[:, 2], np.ones(255)/255, rtol=1e-04, atol=1e-05)
    
    def test_joint_count(self):
        img = np.ones((10, 10, 3), dtype=np.uint8)
        
        hist = color_histogram(img, bins=10, distribution='joint', ranges=((0, 1),(0, 1),(0, 1)), normalization=False)
        
        self.assertEqual(10 * 10 * 3, np.sum(hist))
    
    def test_joint_basic_2_channels(self):
        img = np.ones((10, 10, 3), dtype=np.uint8)
        
        hist = color_histogram(img[:,:, :2], bins=10, distribution='joint', ranges=((0, 1),(0, 1)))
        
        expected = np.zeros((100,), dtype=np.float32)
        expected[99] = 1.0
        
        assert_allclose(hist, expected, rtol=1e-04, atol=1e-05)
    
    def test_joint_basic_3_channels(self):
        img = np.ones((10, 10, 3), dtype=np.uint8)
        
        hist = color_histogram(img, bins=10, distribution='joint', ranges=((0, 1),(0, 1), (0, 1)))
        
        expected = np.zeros((3, 100), dtype=np.float32)
        expected[:, 99] = 1.0
        
        hist = hist.reshape((3, 100))
        
        assert_allclose(hist, expected, rtol=1e-04, atol=1e-05)

