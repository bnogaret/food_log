"""
File including global constants.

For the dataset information:
- `UEC-FOOD 256 <http://foodcam.mobi/dataset256.html>`_
- `UEC-FOOD 100 <http://foodcam.mobi/dataset100.html>`_

For CNN:
- `how to download a pre-trained model <https://github.com/BVLC/caffe/wiki/Model-Zoo>`_
- `github of the segmentation model <https://gist.github.com/jimmie33/339fd0a938ed026692267a60b44c0c58>`_
- `github of the descriptor model <https://gist.github.com/ksimonyan/3785162f95cd2d5fee77>`_
- `list of models <https://github.com/BVLC/caffe/wiki/Model-Zoo#berkeley-trained-models>`_

.. warning::
    Change the values according to your project.

Examples
--------
>>> import constants as const
>>> print(const.PATH_TO_UEFCFOOD256)
/home/nogaret/.virtualenvs/thesis/project/data/UECFOOD256
"""

import os
import numpy as np


__all__ = ['PATH_TO_ROOT_UECFOOD256',
           'PATH_TO_ROOT_UECFOOD100',
           'PATH_TO_IMAGE_DIR',
           'PICKLE_FILENAME_256_GT_BBOX',
           'PICKLE_FILENAME_100_GT_BBOX',
           'PATH_TO_SEG_MODEL_DEF',
           'PATH_TO_SEG_MODEL_WEIGHTS',
           'PATH_TO_SEG_BBOX',
           'PATH_TO_DESCRI_MODEL_DEF',
           'PATH_TO_DESCRI_MODEL_WEIGHTS',
           'MEAN_BGR_VALUES',
           'IMAGE_SIZE'
           ]


PATH_CURRENT_DIRECTORY = os.path.dirname(__file__)
PATH_TO_ROOT_UECFOOD256 = os.path.abspath(PATH_CURRENT_DIRECTORY + "/../data/UECFOOD256/")
"""
str: Path to the root directory including the UEC-FOOD 256 dataset.
"""

PATH_TO_ROOT_UECFOOD100 = os.path.abspath(PATH_CURRENT_DIRECTORY + "/../data/UECFOOD100/")
"""
str: Path to the root directory including the UEC-FOOD 100 dataset.
"""

PATH_TO_IMAGE_DIR = os.path.abspath(PATH_CURRENT_DIRECTORY + "/../img/")
"""
str: Path to the directory to save images (plots ...)
"""

PICKLE_FILENAME_256_GT_BBOX = "256_gt_bbox"
"""
str: filename (doesn't include the path) to the pickle file containing the
ground truth bbox coordinates for UECFOOD 256.
"""

PICKLE_FILENAME_100_GT_BBOX = "100_gt_bbox"
"""
str: filename (doesn't include the path) to the pickle file containing the
ground truth bbox coordinates for UECFOOD 100.
"""

CAFFE_ROOT = "/scratch/s242635/caffe/caffe/"
PATH_TO_SEGMENTATION_MODEL = CAFFE_ROOT + "/models/339fd0a938ed026692267a60b44c0c58/"

PATH_TO_SEG_MODEL_DEF = os.path.abspath(PATH_TO_SEGMENTATION_MODEL + 'deploy.prototxt')
"""
str: Path to the file including the definition of the CNN model.
It is used for segmentation.
"""

PATH_TO_SEG_MODEL_WEIGHTS = os.path.abspath(PATH_TO_SEGMENTATION_MODEL + 'GoogleNet_SOD_finetune.caffemodel')
"""
str: Path to the file including the pre-trained CNN's weights.
It is used for segmentation.
"""

PATH_TO_SEG_BBOX = os.path.abspath(PATH_TO_SEGMENTATION_MODEL + 'center100.txt')
"""
str: Path to the bounding boxes' coordinates.
It is used for segmentation.
"""

PATH_TO_DESCRIPTOR_MODEL = CAFFE_ROOT + "/models/3785162f95cd2d5fee77/"

PATH_TO_DESCRI_MODEL_DEF = os.path.abspath(PATH_TO_DESCRIPTOR_MODEL + 'VGG_ILSVRC_19_layers_deploy.prototxt')
"""
str: Path to the file including the definition of the CNN model.
It is used for classification as a feature descriptor.
"""

PATH_TO_DESCRI_MODEL_WEIGHTS = os.path.abspath(PATH_TO_DESCRIPTOR_MODEL + 'VGG_ILSVRC_19_layers.caffemodel')
"""
str: Path to the file including the pre-trained CNN's weights.
It is used for classification as a feature descriptor.
"""

MEAN_BGR_VALUES = np.asarray([103.939, 116.779, 123.68])
"""
:class:`np.ndarray`: Mean value of the pixels for the BGR channel (defined in this order).

It corresponds to the mean pixel value of the VGG dataset.
"""

IMAGE_SIZE = (224, 224)
"""
tuple: Image size of the input data of the CNN used for the segmentation.
"""

