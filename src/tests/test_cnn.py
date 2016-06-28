import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import unittest
import numpy as np

import caffe

import constants as const
from thesis_lib.io import load_img_as_float
from thesis_lib.cnn import get_transformer_rgb_image, transform_rgb_image


class TransformRgbTest(unittest.TestCase):
    def test_same_behaviour_between_transform(self):
        imagePath = "test_image.jpg"
        image = load_img_as_float(imagePath)
        
        mean_bgr = np.asarray([103.939, 116.779, 123.68])
        
        transformer = get_transformer_rgb_image((1, 3, 224, 224), mean_bgr)
        transformed_image = transformer.preprocess('data', image)
        
        img = transform_rgb_image(image, (224, 224), mean_bgr)
        
        print(np.allclose(img, transformed_image, rtol=1e-04, atol=1e-05))
