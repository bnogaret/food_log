#! /usr/bin/env python3
import os
import glob

# Environment variable to keep glog quiet (use by caffe)
# 0 - debug
# 1 - info
# 2 - warnings
# 3 - errors
os.environ['GLOG_minloglevel'] = '2'

import numpy as np
from skimage.io import imread

from thesis_lib.segmenter import CnnSegmenter
from thesis_lib.io import load_object
from thesis_lib.bbox import get_accuracy_bbox, get_coordinate_resized_rectangles
from thesis_lib.cnn import set_caffe_mode
import constants as const


def get_gt_bbox_for_image(df, path_image):
    """
    example of image_name: "4004"
    """
    # Get the row of the current file
    bbox = df.loc[(df._abs_path == path_image)]
    
    # print(bbox)
    
    if bbox._multi_item.all(axis=0):
        # Case when the same image has multi-food items - multi food labels (so different absolute path)
        image_name = int(os.path.basename(path_image).replace(".jpg", ''))
        bbox = df.loc[(df._img_name == image_name) & (df._multi_item == True)]
    
    return bbox.as_matrix(["_x1", "_y1", "_x2", "_y2"])


def segmentate(root_directory, segmenter, pickle_filename):
    df = load_object(pickle_filename)
    metrics = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    count = 0
    
    # for entry in list(os.scandir(root_directory))[0:5]:
    for entry in os.scandir(root_directory):
        if entry.is_dir(follow_symlinks=False):
            label = int(entry.name)
            
            print(label)
            
            # for path_image in list(glob.iglob(entry.path + '/*.jpg', recursive=False))[0:50]:
            for path_image in glob.iglob(entry.path + '/*.jpg', recursive=False):
                # filename_without_jpg = int(os.path.basename(path_image).replace(".jpg", ''))
                # print(path_image)
                gt_bboxes = get_gt_bbox_for_image(df, path_image)
                
                image = imread(path_image)
                
                resized_ground_truth = get_coordinate_resized_rectangles(image.shape[0:2], const.IMAGE_SIZE, gt_bboxes)
                pred_bboxes = segmenter.get_bbox(image)
                
                # print(resized_ground_truth)
                # print(pred_bboxes)
                
                pred_bboxes, res = get_accuracy_bbox(resized_ground_truth, pred_bboxes)
                
                metrics += res
                count += 1
                
    print(metrics / count)
    

def main():
    segmenter = CnnSegmenter(const.PATH_TO_SEG_MODEL_DEF,
                             const.PATH_TO_SEG_MODEL_WEIGHTS,
                             const.PATH_TO_SEG_BBOX,
                             const.MEAN_BGR_VALUES,
                             const.IMAGE_SIZE,
                             0.01,
                             0.3)
    
    set_caffe_mode()
    
    segmentate(const.PATH_TO_ROOT_UECFOOD256, segmenter, const.PICKLE_FILENAME_256_GT_BBOX)


if __name__ == "__main__":
    main()
