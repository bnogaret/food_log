#! /usr/bin/env python3
import os
import glob
import gc

import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.preprocessing import scale
from skimage.io import imread
from skimage.transform import resize

from thesis_lib.uecfood import read_bb_info_txt
from thesis_lib.transform import get_sub_image_from_rectangle
from thesis_lib.cross_validation import cross_val_multiple_scores
from thesis_lib.io import save_object
from thesis_lib.descriptor import BagOfWordsDescriptor

import constants as const


# Inspiration: https://github.com/amueller/segmentation/blob/master/bow.py


def main():
    VOCABULARY_SIZE = 1000
    STEP_SIZE = 4
    bow = BagOfWordsDescriptor(const.IMAGE_SIZE, VOCABULARY_SIZE, STEP_SIZE, scale_data=False)

    data = []
    target = []

    # for entry in list(os.scandir(const.PATH_TO_ROOT_UECFOOD256))[0:4]:
    for entry in os.scandir(const.PATH_TO_ROOT_UECFOOD256):
        if entry.is_dir(follow_symlinks=False):
            bb_info = []
            read_bb_info_txt(entry.path + "/bb_info.txt", bb_info)
            df = pd.DataFrame(bb_info, columns=['_img_name', '_x1', '_y1', '_x2', '_y2', '_cat', '_abs_path'])

            label = int(entry.name)

            print(label)

            # for image_path in list(glob.iglob(entry.path + '/*.jpg', recursive=False))[0:25]:
            for image_path in glob.iglob(entry.path + '/*.jpg', recursive=False):
                filename_without_jpg = int(os.path.basename(image_path).replace(".jpg", ''))
                gt_bboxes = df.loc[df._img_name == filename_without_jpg].as_matrix(["_x1", "_y1", "_x2", "_y2"])

                image = imread(image_path)

                for bbox in gt_bboxes:
                    # print(bbox)

                    sub_image = get_sub_image_from_rectangle(image, bbox, True)
                    sub_image = resize(sub_image, const.IMAGE_SIZE)

                    data.append(bow.get_feature(sub_image))
                    target.append(label)

    print(len(data), len(target))

    X, y = bow.post_process_data(data, target)

    print("X (type: %s) shape: %s || target (type: %s) shape: %s" % (X.dtype, X.shape, y.dtype, y.shape))

    # "Free memory" to avoid MemoryError
    data = []
    bow = []
    target = []
    print("gc.collect() = ", gc.collect())

    chi2 = AdditiveChi2Sampler(sample_steps=2)

    X = chi2.fit_transform(X)
    X = scale(X)

    print("X (type: %s) shape: %s || target (type: %s) shape: %s" % (X.dtype, X.shape, y.dtype, y.shape))
    classifier = LinearSVC(fit_intercept=False, dual=False)

    print(classifier)

    cv_scores = cross_val_multiple_scores(classifier,
                                          X=X,
                                          y=y,
                                          n_folds=10,
                                          n_jobs=1)

    print(cv_scores)

    save_object(cv_scores['cv_confusion_matrix'],
                "cm_bow",
                overwrite=True)


if __name__ == "__main__":
    main()

