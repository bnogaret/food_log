#! /usr/bin/env python3
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from skimage.color import rgb2hsv
from skimage.io import imread

import init_path
import constants as const
from thesis_lib.histogram import color_histogram


def argument_parser():
    parser = argparse.ArgumentParser(description="Graph for report")

    parser.add_argument('-g',
                        '--graph',
                        help=('Which graph to generate: ' +
                             '"imagenet", "segoverlap", "obesity", "jointhistogram"'),
                        choices=['imagenet','segoverlap','obesity','jointhistogram'],
                        default='imagenet',
                        type=str)
    args = parser.parse_args()
    return args

def graph_imagenet():
    """
    From: https://arxiv.org/pdf/1409.0575v3.pdf
    """
    x = np.arange(2010, 2015, dtype=np.int)
    classification_error = np.asarray([28.2, 25.8, 16.4, 11.7, 6.7])
    localization_error = np.asarray([np.nan, 42.5, 34.2, 30, 25.3])

    plt.figure(figsize=(6, 6))
    fig, ax = plt.subplots()
    ax.plot(x, classification_error, label="Average classification error")
    ax.plot(x, localization_error, label="Average localization error")
    ax.set_xlim(2010, 2014)
    ax.get_xaxis().set_major_locator(mticker.MaxNLocator(integer=True))
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.legend()
    plt.title("Image Net's error result")

    plt.savefig(const.PATH_TO_IMAGE_DIR + "/imagenet.jpg")

def graph_segmentation_overlap():
    x = np.arange(0.5, 1.01, 0.05)
    accuracy = np.asarray([0.73, 0.69, 0.65, 0.60, 0.55, 0.47, 0.36, 0.24, 0.10, 0.01, 0.0])
    precision = np.asarray([0.74, 0.70, 0.66, 0.61, 0.55, 0.47, 0.36, 0.24, 0.10, 0.01, 0.0])
    recall = np.asarray([0.79, 0.75, 0.70, 0.61, 0.55, 0.47, 0.36, 0.24, 0.10, 0.01, 0.0])

    fig = plt.figure(figsize=(6, 14))

    ax = plt.subplot(311)
    ax.plot(x, accuracy)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_title("Average accuracy")

    ax = plt.subplot(312)
    ax.plot(x, precision)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_title("Average precision")

    ax = plt.subplot(313)
    ax.plot(x, recall)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_title("Average recall")

    plt.savefig(const.PATH_TO_IMAGE_DIR + "/segmentation_overlap.jpg")


def graph_joint_histogram():
    path = const.PATH_TO_ROOT_UECFOOD256 + "/1/10.jpg"
    image = imread(path)

    hsv = rgb2hsv(image)
    ch = color_histogram(hsv[:,:,:2], 20, ranges=((0,1),(0,1)), distribution="joint", normalization=False)
    ch = ch.reshape((20, 20))

    cax = plt.matshow(ch, interpolation='nearest')
    plt.title("Joint histogram for Hue and Saturation", y=1.08)

    plt.colorbar(cax)
    plt.savefig(const.PATH_TO_IMAGE_DIR + "/joint_histogram.jpg")


def graph_obesity():
    """
    https://www.theguardian.com/society/2015/may/05/obesity-crisis-projections-uk-2030-men-women
    https://www.theguardian.com/news/datablog/2014/may/29/how-obese-is-the-uk-obesity-rates-compare-other-countries
    """
    x = np.asarray([1980, 1990, 2000, 2013, 2030])
    # obesity: UK: increase of 12 % between 1980 and 2013 (BMI > 30)
    obesity = np.asarray([0.14, 0.14, 0.20, 0.25, np.nan])
    # overweight: UK: increase of 13 % between 1980 and 2013 (BMI  > 20)
    overweight = np.asarray([0.49, 0.51, 0.59, 0.62, np.nan])
    # prediction
    obesity_pred = np.asarray([np.nan, np.nan, np.nan, 0.25, 0.35])
    overweight_pred = np.asarray([np.nan, np.nan, np.nan, 0.62, 0.69])

    plt.figure(figsize=(6, 6))
    fig, ax = plt.subplots()
    ax.plot(x, obesity, label="Obese", color='g', linewidth=2)
    ax.plot(x, overweight, label="Overweight (include obese)", color='r', linewidth=2)
    ax.plot(x, obesity_pred, marker='o', linestyle='--', color='g', linewidth=2)
    ax.plot(x, overweight_pred, marker='o', linestyle='--', color='r', linewidth=2)
    ax.set_xlim(1980, 2030)
    ax.get_xaxis().set_major_locator(mticker.MaxNLocator(integer=True))
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.set_ylim(0.0, 0.7)
    plt.legend(loc=4)
    plt.title("Obesity and overweight rates in the UK and forecast for 2030")

    plt.savefig(const.PATH_TO_IMAGE_DIR + "/obesity_uk.jpg")


def main():
    args = argument_parser()

    print(args)

    if args.graph == "imagenet":
        graph_imagenet()
    elif args.graph == "segoverlap":
        graph_segmentation_overlap()
    elif args.graph == "jointhistogram":
        graph_joint_histogram()
    elif args.graph == "obesity":
        graph_obesity()

if __name__ == "__main__":
    main()
