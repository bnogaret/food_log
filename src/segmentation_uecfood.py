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
from thesis_lib.bbox import get_correct_bbox
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
        # Case when the same image has multi-food items that can be from different labels (so different absolute path)
        image_name = int(os.path.basename(path_image).replace(".jpg", ''))
        bbox = df.loc[(df._img_name == image_name) & (df._multi_item == True)]
    
    return bbox.as_matrix(["_x1", "_y1", "_x2", "_y2", "_cat"])


def segmentate(segmenter, overlap_for_correctness, root_directory, pickle_filename):
    df = load_object(pickle_filename)
    
    for ofc in overlap_for_correctness:
        metrics = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        count = 0
        
        # for entry in list(os.scandir(root_directory))[0:2]:
        for entry in os.scandir(root_directory):
            if entry.is_dir(follow_symlinks=False):
                label = int(entry.name)
                
                # print(label)
                
                # for path_image in list(glob.iglob(entry.path + '/*.jpg', recursive=False))[:20]:
                for path_image in glob.iglob(entry.path + '/*.jpg', recursive=False):
                    # Retrieve ground truth bbox for the image
                    gt_bboxes = get_gt_bbox_for_image(df, path_image)
                    
                    image = imread(path_image)
                    
                    pred_bboxes = segmenter.get_bbox(image)
                    
                    pred_bboxes, gt_bboxes, metric_res = get_correct_bbox(gt_bboxes, pred_bboxes, threshold=ofc)
                    
                    metrics += metric_res
                    count += 1
        
        print("Overlap for correctness = ", ofc)
        print("Metrics: ", metrics / count)


def main():
    segmenter = CnnSegmenter(const.PATH_TO_SEG_MODEL_DEF,
                             const.PATH_TO_SEG_MODEL_WEIGHTS,
                             const.PATH_TO_SEG_BBOX,
                             const.MEAN_BGR_VALUES,
                             const.IMAGE_SIZE,
                             0.01,
                             0.3)
    
    set_caffe_mode()
    
    # overlap_for_correctness = [1.0]
    overlap_for_correctness = np.arange(0.5, 1.01, 0.05)
    
    segmentate(segmenter, overlap_for_correctness, const.PATH_TO_ROOT_UECFOOD256, const.PICKLE_FILENAME_256_GT_BBOX)


if __name__ == "__main__":
    main()

