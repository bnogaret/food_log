#! /usr/bin/env python3
import os
import argparse
import glob

# Environment variable to keep glog quiet (use by caffe)
# 0 - debug
# 1 - info
# 2 - warnings
# 3 - errors
os.environ['GLOG_minloglevel'] = '2'

import caffe
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from skimage.transform import resize
from skimage.io import imread

import constants as const
from thesis_lib.uecfood import read_bb_info_txt
from thesis_lib.io import save_object
from thesis_lib.transform import get_sub_image_from_rectangle
from thesis_lib.cross_validation import cross_val_multiple_scores
from thesis_lib.descriptor import HistogramDescriptor, CnnDescriptor
from thesis_lib.cnn import set_caffe_mode


# TODO: add bow


def argument_parser():
    parser = argparse.ArgumentParser(description="Classification phase comparaison (either classifier or feature comparison)")
    parser.add_argument('-f',
                        '--feature',
                        help='Choose a feature descriptor between: ' + \
                             '"cnn": using last layer of a convolutional neural network ' + \
                             '"bow": bag of wisual words ' + \
                             '"rgb": local binary pattern and marginal histogram ' + \
                             '"hxv": local binary pattern and joint histogram on hs',
                        choices=['cnn', 'bow', 'rgb', 'hsv'],
                        default='hsv',
                        type=str)
    
    parser.add_argument('-c',
                        '--classifier',
                        help='Choose a classifier between : ' + \
                             '"sgd": SGD classifier' + \
                             '"rf": random forest ' + \
                             '"tree": decision tree ' + \
                             '"knn": k-nearest neighborhood ' + \
                             '"nb": naive bayes ' + \
                             '"svm": linear svm ' + \
                             '"all": all the possible classifiers',
                        choices=['sgd', 'rf', 'tree', 'knn', 'nb', 'svm','all'],
                        default='sgd',
                        type=str)
    
    parser.add_argument('-d',
                        '--dataset',
                        help='Choose the dataset between : ' + \
                             '"100": UEC-FOOD100 ' + \
                             '"256": UEC-FOOD256',
                        choices=['256', '100'],
                        default='100',
                        type=str)
    
    args = parser.parse_args()
    return args

def classify(root_directory, descriptor, classifiers):
    data = []
    target = []    
    
    for entry in list(os.scandir(root_directory))[0:6]:
    # for entry in os.scandir(root_directory):
        if entry.is_dir(follow_symlinks=False):
            bb_info = []
            read_bb_info_txt(entry.path + "/bb_info.txt", bb_info)
            df = pd.DataFrame(bb_info, columns=['_img_name', '_x1', '_y1', '_x2', '_y2', '_cat', '_abs_path'])
            
            label = entry.name
            
            print(label)
            
            for image_path in list(glob.iglob(entry.path + '/*.jpg', recursive=False))[0:25]:
            # for image_path in glob.iglob(entry.path + '/*.jpg', recursive=False):
                # print(image_path)
                filename_without_jpg = int(os.path.basename(image_path).replace(".jpg", ''))
                gt_bboxes = df.loc[df._img_name == filename_without_jpg].as_matrix(["_x1", "_y1", "_x2", "_y2"])
                
                image = imread(image_path)
                
                for bbox in gt_bboxes:
                    # print(bbox)
                    
                    sub_image = get_sub_image_from_rectangle(image, bbox, True)
                    sub_image = resize(sub_image, const.IMAGE_SIZE)
                    
                    data.append(descriptor.get_feature(sub_image))
                    target.append(label)
    
    X, y = descriptor.post_process_data(data, target)
    
    CATEGORY_FILE = root_directory + "/category.txt"

    df = get_name_and_category(CATEGORY_FILE)
    
    for classifier in classifiers:
        print(classifier)
        cv_scores = cross_val_multiple_scores(classifier,
                                              X=X,
                                              y=y,
                                              n_folds=5,
                                              n_jobs=4)
        
        print(cv_scores)
        save_object(cv_scores['cv_confusion_matrix'],
                    "cm_" + classifier.__class__.__name__,
                    overwrite=True)


def main():
    args = argument_parser()
    
    print(args)
    
    set_caffe_mode()
    
    arg_dataset = {
        '256'   : const.PATH_TO_ROOT_UECFOOD256,
        '100'   : const.PATH_TO_ROOT_UECFOOD100,
    }
    
    arg_feature = {
        'hsv'   : HistogramDescriptor(),
        'rgb'   : HistogramDescriptor(bin_ch = 50, distribution='marginal'),
        'cnn'   : CnnDescriptor('fc7',
                                const.PATH_TO_DESCRI_MODEL_DEF,
                                const.PATH_TO_DESCRI_MODEL_WEIGHTS,
                                const.MEAN_BGR_VALUES,
                                const.IMAGE_SIZE)
    }
    
    arg_classifier = {
        'sgd'   : SGDClassifier(n_iter=250, penalty='none', n_jobs=4),
        'rf'    : RandomForestClassifier(n_estimators=250, min_samples_leaf=20, n_jobs=4),
        'tree'  : DecisionTreeClassifier(min_samples_leaf=10),
        'knn'   : KNeighborsClassifier(n_neighbors=10, leaf_size=30),
        'nb'    : GaussianNB(),
        'svm'   : LinearSVC(fit_intercept=False, dual=False)
    }
    
    if args.classifier == "all":
        classifiers = arg_classifier.values()
    else:
        classifiers = [arg_classifier[args.classifier]]
    
    classify(arg_dataset[args.dataset],
             arg_feature[args.feature],
             classifiers)
    

if __name__ == "__main__":
    main()
