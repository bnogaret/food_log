"""
File including global constants.

For the dataset information: 
- `UEC FOOD 256 <http://foodcam.mobi/dataset256.html>`_

For the segmentation model:
- `how to download a pre-trained model <https://github.com/BVLC/caffe/wiki/Model-Zoo>`_
- `github of the segmentation model <https://gist.github.com/jimmie33/339fd0a938ed026692267a60b44c0c58>`_
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
           'PATH_TO_SEG_MODEL_DEF',
           'PATH_TO_SEG_MODEL_WEIGHTS',
           'PATH_TO_SEG_BBOX',
           'MEAN_BGR_VALUE',
           'IMAGE_SIZE'
           ]


PATH_CURRENT_DIRECTORY = os.path.dirname(__file__)
PATH_TO_ROOT_UECFOOD256 = os.path.abspath(PATH_CURRENT_DIRECTORY + "/../data/UECFOOD256/")
"""
str: Path to the root directory including the UEC FOOD 256 dataset.
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
It used for segmentation.
"""

PATH_TO_SEG_BBOX = os.path.abspath(PATH_TO_SEGMENTATION_MODEL + 'center100.txt')
"""
str: Path to the bounding boxes' coordinates.
It is used for segmentation.
"""

MEAN_BGR_VALUE = np.asarray([103.939, 116.779, 123.68])
"""
:class:`np.ndarray`: Mean value of the pixels for the BGR channel (defined in this order).

It corresponds to the mean pixel value of the VGG dataset.
"""

IMAGE_SIZE = (224, 224)
"""
tuple: Image size of the input data of the CNN used for the segmentation.

"""

