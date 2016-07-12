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

import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_predict, train_test_split, ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

from skimage.transform import resize
from skimage.io import imread

import constants as const
from thesis_lib.uecfood import read_bb_info_txt
from thesis_lib.io import save_object
from thesis_lib.transform import get_sub_image_from_rectangle
from thesis_lib.cross_validation import cross_val_multiple_scores
from thesis_lib.descriptor import HistogramMomentDescriptor, CnnDescriptor
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
                             '"ada": Adaboost classifier ' + \
                             '"gb": Gradient Boosting ' + \
                             '"all": all the possible classifiers',
                        choices=['sgd', 'rf', 'tree', 'knn', 'nb', 'svm', 'ada', 'gb', 'all'],
                        default='sgd',
                        type=str)

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

def classify(root_directory, descriptor, classifiers, param_grid):
    data = []
    target = []

    for entry in list(os.scandir(root_directory))[0:4]:
    # for entry in os.scandir(root_directory):
        if entry.is_dir(follow_symlinks=False):
            bb_info = []
            read_bb_info_txt(entry.path + "/bb_info.txt", bb_info)
            df = pd.DataFrame(bb_info, columns=['_img_name', '_x1', '_y1', '_x2', '_y2', '_cat', '_abs_path'])

            label = int(entry.name)

            print(label)

            for image_path in list(glob.iglob(entry.path + '/*.jpg', recursive=False))[0:40]:
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

    for classifier in classifiers:
        print(classifier)
        print(param_grid)

        if param_grid is not None:
            if hasattr(classifier, 'n_estimators'):
                classifier.n_estimators = 25 # For faster search!

            X_validation, X, y_validation, y = train_test_split(X, y, train_size=0.10)

            clf = GridSearchCV(classifier,
                               param_grid,
                               cv=5,
                               scoring='accuracy',
                               n_jobs=-1,
                               refit=True)

            clf.fit(X_validation, y_validation)

            print("Best parameters set found on development set:")
            print(clf.best_params_)

            classifier = clf.best_estimator_

            if hasattr(classifier, 'n_estimators'):
                classifier.n_estimators = 500

            print(classifier)

        y_pred = cross_val_predict(classifier, X, y, n_jobs=4, cv=10)

        print(classification_report(y, y_pred))
        cm = confusion_matrix(y, y_pred)
        print(cm)

        """
        save_object(cm,
                    "cm_" + descriptor.__class__.__name__ + "_" + classifier.__class__.__name__,
                    overwrite=True)
        """
        """
        cv_scores = cross_val_multiple_scores(classifier,
                                              X=X,
                                              y=y,
                                              n_folds=10,
                                              n_jobs=4)
        print(cv_scores)
        save_object(cv_scores['cv_confusion_matrix'],
                    "cm_" + classifier.__class__.__name__,
                    overwrite=True)
        """


def main():
    args = argument_parser()

    print(args)

    set_caffe_mode()

    arg_dataset = {
        '256'   : const.PATH_TO_ROOT_UECFOOD256,
        '100'   : const.PATH_TO_ROOT_UECFOOD100,
    }

    arg_feature = {
        'hsv'   : HistogramMomentDescriptor(bin_lbp=98, bin_ch = 30, distribution='joint', scale_data=True),
        'rgb'   : HistogramMomentDescriptor(bin_lbp=98, bin_ch = 100, distribution='marginal', scale_data=True),
        'cnn'   : CnnDescriptor('fc7',
                                const.PATH_TO_DESCRI_MODEL_DEF,
                                const.PATH_TO_DESCRI_MODEL_WEIGHTS,
                                const.MEAN_BGR_VALUES,
                                const.IMAGE_SIZE,
                                scale_data=False)
    }

    arg_classifier = {
        'sgd'   : SGDClassifier(n_iter=500, penalty='none', n_jobs=2),
        'rf'    : RandomForestClassifier(n_estimators=500, n_jobs=2),
        'tree'  : DecisionTreeClassifier(min_samples_leaf=10),
        'knn'   : KNeighborsClassifier(n_neighbors=10, leaf_size=30, n_jobs=2),
        'nb'    : GaussianNB(),
        'svm'   : LinearSVC(fit_intercept=False, dual=False),
        'ada'   : AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=10), n_estimators=200),
        'gb'    : GradientBoostingClassifier(n_estimators=10)
    }

    arg_tuned = {
        'sgd'   : [{'penalty' : ['l2', 'none'],
                   'alpha' : [1e-2, 1e-3, 1e-4, 1e-5],
                   'loss' : ['hinge', 'log']},
                   {'penalty' : ['elasticnet'],
                   'alpha' : [1e-2, 1e-3, 1e-4, 1e-5],
                   'loss' : ['hinge', 'log'],
                   'l1_ratio' : [0.15, 0.25]}],
        'rf'    : {'max_depth': [5, None],
                   'max_features': ['sqrt', 'log2'],
                   'min_samples_split': [1, 10, 20],
                   'min_samples_leaf': [10, 20, 30],
                   'criterion': ['gini', 'entropy']},
        'tree'  : {'max_features': ['sqrt', 'log2'],
                   'min_samples_split': [10, 20],
                   'min_samples_leaf': [10, 20],
                   'criterion': ['gini', 'entropy']},
        'knn'   : {'n_neighbors': [15, 10, 5],
                   'weights': ['uniform', 'distance'],
                   'metric': ['minkowski','euclidean','manhattan'],
                   'leaf_size': [50, 30, 10]},
        'nb'    : None,
        'svm'   : {'penalty' : ['l2'],
                   'C' : [10, 1, 0.1],
                   'loss' : ['hinge', 'squared_hinge'],
                   'multi_class' : ['ovr', 'crammer_singer'],
                   'dual' : [True]},
        'ada'   : None,
        'gb'    : {'loss': ['deviance', 'exponential'],
                   'max_features': ['sqrt', 'log2'],
                   'min_samples_split': [10, 20],
                   'min_samples_leaf': [10, 20]}
    }

    if args.classifier == 'all':
        classifiers = arg_classifier.values()
        param_grid = None
    else:
        classifiers = [arg_classifier[args.classifier]]
        param_grid = arg_tuned[args.classifier]

    classify(arg_dataset[args.dataset],
             arg_feature[args.feature],
             classifiers,
             param_grid)


if __name__ == "__main__":
    main()
