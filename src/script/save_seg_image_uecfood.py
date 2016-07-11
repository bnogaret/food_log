#! /usr/bin/env python3
import os
import glob
import argparse

# Environment variable to keep glog quiet (use by caffe)
# 0 - debug
# 1 - info
# 2 - warnings
# 3 - errors
os.environ['GLOG_minloglevel'] = '2'

import timeit

import caffe
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import resize

import init_path

from thesis_lib.draw import draw_bounding_boxes, display_arrays
from thesis_lib.bbox import *
from thesis_lib.cnn import *
from thesis_lib.io import load_object
from thesis_lib.segmenter import CnnSegmenter

import constants as const


def argument_parser():
    parser = argparse.ArgumentParser(description="Save image of segmentation")
    parser.add_argument('-f',
                        '--filename',
                        help='Filename of the image (default: 97)',
                        default=97,
                        type=int)
    parser.add_argument('-tn',
                        '--threshold-net',
                        help='Threshold of confidence for the CNN (default: 0.01)',
                        default = 0.01,
                        type=float)
    parser.add_argument('-to',
                        '--threshold-overlap',
                        help='Threshold of overlap for the non-maxima suppression (default: 0.5)',
                        default = 0.5,
                        type=float)
    args = parser.parse_args()
    return args


def find_image_path(img_name):
    for filename in glob.iglob(const.PATH_TO_ROOT_UECFOOD256 + '/**/' + str(img_name) + '.jpg', recursive=True):
        return filename


def get_ground_truth_bbox(path_image, initial_image_size):
    # Load the pickle save dataframe containing 7 columns:
    # '_img_name' (int) : image name (exclusing the '.jpg')
    # '_x1' (int), '_y1' (int) : coordinate of one of the rectangle point
    # '_x2' (int), '_y2' (int) : coordinate of the opposite rectangle corner
    # '_cat' (int) : class of the bbox
    # '_abs_path' (str) : path to the file
    # _multi_item (bool) : whether several categories are on this picture
    abs_path = os.path.abspath(path_image)
    df = load_object(const.PICKLE_FILENAME_256_GT_BBOX)

    # Get the row of the current file
    bbox = df.loc[(df._abs_path == abs_path)]
    
    # print(bbox)
    
    if bbox._multi_item.all(axis=0):
        # Case when the same image has multi-food items - multi food labels (so different absolute path)
        image_name = int(os.path.basename(abs_path).replace(".jpg", ''))
        bbox = df.loc[(df._img_name == image_name) & (df._multi_item == True)]
    
    gt_bboxes = bbox.as_matrix(["_x1", "_y1", "_x2", "_y2"])
    resized_ground_truth = get_coordinate_resized_rectangles(initial_image_size, const.IMAGE_SIZE, gt_bboxes)

    return resized_ground_truth


def save_segmentation_image(img_name, threshold_net, threshold_overlap):
    path_img = find_image_path(img_name)
    if path_img is None:
        raise Exception("Unknown image id provided")
    
    path_img = os.path.abspath(path_img)
    print(path_img)
    
    # In my case, I'm using the cpu. To change if a GPU is enable.
    set_caffe_mode()
    
    image = imread(path_img)

    segmenter = CnnSegmenter(const.PATH_TO_SEG_MODEL_DEF,
                             const.PATH_TO_SEG_MODEL_WEIGHTS,
                             const.PATH_TO_SEG_BBOX,
                             const.MEAN_BGR_VALUES,
                             const.IMAGE_SIZE,
                             threshold_net,
                             threshold_overlap)
    
    boxes = segmenter.get_bbox(image)
    
    # Case where no overlapping_suppresion is done
    segmenter = CnnSegmenter(const.PATH_TO_SEG_MODEL_DEF,
                             const.PATH_TO_SEG_MODEL_WEIGHTS,
                             const.PATH_TO_SEG_BBOX,
                             const.MEAN_BGR_VALUES,
                             const.IMAGE_SIZE,
                             threshold_net,
                             1.0)
    
    predicted_boxes = segmenter.get_bbox(image)

    # Dipslay the output of the first layer (first 4 only)
    feat = segmenter.net.blobs['conv1/7x7_s2'].data[0, :4]
    print(feat.shape)
    display_arrays(feat,
                   const.PATH_TO_IMAGE_DIR + "/first_conv_layer_" + str(img_name) + ".jpg",
                   title="Visualisation of the first convolutional layer's output")
    
    resized_ground_truth = get_ground_truth_bbox(path_img, image.shape[0:2])
    
    resized_picture = resize(image, const.IMAGE_SIZE)
    
    fig = plt.figure(figsize=(6, 14))

    ax = plt.subplot(311)
    ax.imshow(draw_bounding_boxes(resized_picture, resized_ground_truth, color=0, copy=True))
    ax.axis('off')
    ax.set_title("Ground truth boxes")

    ax = plt.subplot(312)
    ax.imshow(draw_bounding_boxes(resized_picture, predicted_boxes, color=0, copy=True))
    ax.axis('off')
    ax.set_title("Predicted boxes before selection")

    ax = plt.subplot(313)
    ax.imshow(draw_bounding_boxes(resized_picture, boxes, color=0, copy=True))
    ax.axis('off')
    ax.set_title("Predicted boxes after selection")

    plt.savefig(const.PATH_TO_IMAGE_DIR + "/seg_" + str(img_name) + ".jpg")


if __name__ == "__main__":
    args = argument_parser()
    print(args)
    save_segmentation_image(args.filename, args.threshold_net, args.threshold_overlap)
