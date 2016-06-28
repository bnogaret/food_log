import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import unittest
import numpy as np

import caffe

import constants as const
from thesis_lib.io import load_img_as_float


class LoadImgTest(unittest.TestCase):
    def test_same_behaviour_as_caffe_for_rgb(self):
        imagePath = "color_image.jpg"
        img_caffe = caffe.io.load_image(imagePath)
        img = load_img_as_float(imagePath)
        
        self.assertTrue(np.allclose(img, img_caffe))
