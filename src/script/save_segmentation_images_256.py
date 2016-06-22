#! /usr/bin/env python3
import os
import glob
import argparse

import caffe
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.transform import resize

import init_path

from thesis_lib.draw import draw_bounding_boxes, draw_arrays
from thesis_lib.bbox import *
from thesis_lib.cnn import *
from thesis_lib.io import load_object


PATH_CURRENT_DIRECTORY = os.path.dirname(__file__)
PATH_TO_UECFOOD256 = PATH_CURRENT_DIRECTORY + "/../../data/UECFOOD256/"
PATH_TO_IMAGE_DIR = PATH_CURRENT_DIRECTORY + "/../../img/"

CAFFE_ROOT = "/scratch/s242635/caffe/caffe/"
PATH_TO_SEGMENTATION_MODEL = CAFFE_ROOT + "/models/339fd0a938ed026692267a60b44c0c58/"

PATH_TO_SEG_MODEL_DEF = PATH_TO_SEGMENTATION_MODEL + 'deploy.prototxt'
PATH_TO_SEG_MODEL_WEIGHTS = PATH_TO_SEGMENTATION_MODEL + 'GoogleNet_SOD_finetune.caffemodel'
PATH_TO_BBOX = PATH_TO_SEGMENTATION_MODEL + 'center100.txt'

IMAGE_SIZE = (224, 224)
MEAN_BGR_VALUE = np.asarray([103.939, 116.779, 123.68])


def argument_parser():
    parser = argparse.ArgumentParser(description="Save image of segmentation")
    parser.add_argument('-i',
                        '--image-id',
                        help='Id of the image (default: 97)',
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


def find_image_path(image_id):
    for filename in glob.iglob(PATH_TO_UECFOOD256 + '/**/' + str(image_id) + '.jpg', recursive=True):
        return filename


def get_ground_truth_bbox(image_id, initial_image_size):
    # Load the pickle save dataframe containing 6 columns:
    # 'image_id' (int) : image name (exclusing the '.jpg')
    # 'x1' (int), 'y1' (int) : coordinate of one of the rectangle point
    # 'x2' (int), 'y2' (int) : coordinate of the opposite rectangle corner
    # 'label' (int) : class of the bbox
    ground_truth_bbox = load_object("256_gt_bbox")
    
    # Get the rows of the current file
    ground_truth = ground_truth_bbox.loc[ground_truth_bbox.image_id == image_id]
    # Extract the numpy array corresponding to the bbox coordinates
    ground_truth_box = ground_truth.as_matrix(["x1", "y1", "x2", "y2"])
    
    resized_ground_truth = get_coordinate_resized_rectangles(initial_image_size,
                                                             IMAGE_SIZE,
                                                             ground_truth_box)
    
    print(ground_truth)
    print(resized_ground_truth)
    
    return resized_ground_truth
    

def save_segmentation_image(image_id, threshold_net, threshold_overlap):
    path_img = find_image_path(image_id)
    if path_img is None:
        raise Exception("Unknown image id provided")
    
    # In my case, I'm using the cpu. To change if a GPU is enable.
    set_caffe_mode()
    
    cnn_bbox_coordinates = np.loadtxt(PATH_TO_BBOX, np.float, delimiter=',')
    
    net = get_trained_network(PATH_TO_SEG_MODEL_DEF, PATH_TO_SEG_MODEL_WEIGHTS, IMAGE_SIZE)
    
    transformer = get_transformer_rgb_image(net.blobs['data'].data.shape, MEAN_BGR_VALUE)
    
    image = caffe.io.load_image(path_img)
    transformed_image = transformer.preprocess('data', image)
    print(image.shape)
    print(transformed_image.shape)
    
    # Feed the data to the network
    net.blobs['data'].data[...] = transformed_image

    output = net.forward()
    prob = output['prob'][0]
    print(prob)
    print(prob.shape)
    
    # Dipslay the output of the first layer (first 4 only)
    feat = net.blobs['conv1/7x7_s2'].data[0, :4]
    print(feat.shape)
    draw_arrays(feat,
                PATH_TO_IMAGE_DIR + "first_conv_layer_" + str(image_id) + ".jpg",
                title="Visualisation of the first convolutional layer's output")
    
    # Select indices superior to a threshold
    predicted_index = np.where(prob > threshold_net)[0]
    if predicted_index.size == 0:
        predicted_index = np.where(prob == prob.max())[0]

    print(cnn_bbox_coordinates)
    print(cnn_bbox_coordinates.shape, cnn_bbox_coordinates.dtype)
    print(cnn_bbox_coordinates[predicted_index])

    # np.ceil: return a float dtype
    predicted_boxes = np.ceil(cnn_bbox_coordinates[predicted_index] * 224)
    resized_picture = resize(image, IMAGE_SIZE)

    boxes = non_maxima_suppression(predicted_boxes, prob[predicted_index], threshold_overlap)
    
    resized_ground_truth = get_ground_truth_bbox(image_id, image.shape[0:2])
    
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
    
    plt.savefig(PATH_TO_IMAGE_DIR + "seg_" + str(image_id) + ".jpg")


if __name__ == "__main__":
    args = argument_parser()
    print(args)
    save_segmentation_image(args.image_id, args.threshold_net, args.threshold_overlap)

