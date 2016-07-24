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

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from skimage.io import imread
from skimage.transform import resize

from thesis_lib.segmenter import CnnSegmenter
from thesis_lib.descriptor import HistogramMomentDescriptor, CnnDescriptor
from thesis_lib.io import load_object, save_object
from thesis_lib.bbox import get_correct_bbox, get_coordinate_resized_rectangles
from thesis_lib.transform import get_sub_image_from_rectangle
from thesis_lib.cnn import set_caffe_mode
import constants as const


def argument_parser():
    parser = argparse.ArgumentParser(description="Whole process: segmentation + classification")
    parser.add_argument('-d',
                        '--dataset',
                        help='Choose the dataset between : ' + \
                             '"100": UEC-FOOD100 ' + \
                             '"256": UEC-FOOD256',
                        choices=['256', '100'],
                        default='256',
                        type=str)
    
    args = parser.parse_args()
    return args


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


def segmentate_classify(segmenter, descriptor, classifier, root_directory, pickle_filename, dataset_name):
    df = load_object(pickle_filename)
    metrics = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    count = 0
    
    target = []
    data = []
    
    # for entry in list(os.scandir(root_directory))[0:2]:
    for entry in os.scandir(root_directory):
        if entry.is_dir(follow_symlinks=False):
            label = int(entry.name)
            
            print(label)
            
            # for path_image in list(glob.iglob(entry.path + '/*.jpg', recursive=False))[:20]:
            for path_image in glob.iglob(entry.path + '/*.jpg', recursive=False):
                # filename_without_jpg = int(os.path.basename(path_image).replace(".jpg", ''))
                # print(path_image)
                
                # coordinates + category
                gt_bboxes = get_gt_bbox_for_image(df, path_image)
                
                image = imread(path_image)
                
                # resized coordinates + category
                # resized_ground_truth = get_coordinate_resized_rectangles(image.shape[0:2], const.IMAGE_SIZE, gt_bboxes)
                pred_bboxes = segmenter.get_bbox(image)
                
                pred_bboxes, gt_bboxes, metric_res = get_correct_bbox(gt_bboxes, pred_bboxes)
                
                metrics += metric_res
                count += 1
                
                for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
                    # print(pred_bbox, gt_bbox)
                    
                    sub_image = get_sub_image_from_rectangle(image, pred_bbox, True)
                    sub_image = resize(sub_image, const.IMAGE_SIZE)
                    
                    data.append(descriptor.get_feature(sub_image))
                    target.append(gt_bbox[4])
    
    X, y = descriptor.post_process_data(data, target)
    
    y_pred = cross_val_predict(classifier, X, y, n_jobs=4, cv=10)

    print(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    print("Metrics: ", metrics / count)

    save_object(cm,
                "cm_" + dataset_name + "_" +  segmenter.__class__.__name__ + "_" + descriptor.__class__.__name__ + "_" + classifier.__class__.__name__,
                overwrite=True)


def main():
    args = argument_parser()
    
    print(args)
    
    arg_dataset = {
        '256'   : (const.PATH_TO_ROOT_UECFOOD256, const.PICKLE_FILENAME_256_GT_BBOX, "256"),
        '100'   : (const.PATH_TO_ROOT_UECFOOD100, const.PICKLE_FILENAME_100_GT_BBOX, "100"),
    }    
    
    set_caffe_mode()
    
    segmenter = CnnSegmenter(const.PATH_TO_SEG_MODEL_DEF,
                             const.PATH_TO_SEG_MODEL_WEIGHTS,
                             const.PATH_TO_SEG_BBOX,
                             const.MEAN_BGR_VALUES,
                             const.IMAGE_SIZE,
                             0.01,
                             0.3)
    
    descriptor = CnnDescriptor('fc7',
                               const.PATH_TO_DESCRI_MODEL_DEF,
                               const.PATH_TO_DESCRI_MODEL_WEIGHTS,
                               const.MEAN_BGR_VALUES,
                               const.IMAGE_SIZE,
                               scale_data=True)
    
    classifier = RandomForestClassifier(n_estimators=500, n_jobs=2, min_samples_split=10, min_samples_leaf=10)
    
    segmentate_classify(segmenter, descriptor, classifier, arg_dataset[args.dataset][0], arg_dataset[args.dataset][1], arg_dataset[args.dataset][2])


if __name__ == "__main__":
    main()

