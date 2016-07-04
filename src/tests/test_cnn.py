import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import unittest
import numpy as np
from numpy.testing import assert_allclose

import caffe

import constants as const
from thesis_lib.io import load_img_as_float
from thesis_lib.cnn import get_transformer_rgb_image, transform_rgb_image
from skimage.io import imread
from skimage import img_as_float


class TransformRgbTest(unittest.TestCase):
    def test_same_behaviour_between_transform(self):
        image_path = "color_image.jpg"
        image = load_img_as_float(image_path)
        
        mean_bgr = np.asarray([103.939, 116.779, 123.68])
        
        transformer = get_transformer_rgb_image((1, 3, 224, 224), mean_bgr)
        transformed_image = transformer.preprocess('data', image)
        
        img = transform_rgb_image(image, (224, 224), mean_bgr)
        
        assert_allclose(img, transformed_image, rtol=1e-04, atol=1e-05)
    
    def test_same_behaviour_between_load_img_as_float(self):
        image_path = "color_image.jpg"
        image = load_img_as_float(image_path)
        img = caffe.io.load_image(image_path)
        
        assert_allclose(img, image, rtol=1e-04, atol=1e-05)
        
    def test_same_behaviour_between_load_image(self):
        image_path = "color_image.jpg"
        img = caffe.io.load_image(image_path)
        
        image = imread(image_path)
        image = img_as_float(image).astype(np.float32)
        
        assert_allclose(img, image, rtol=1e-04, atol=1e-05)
    
