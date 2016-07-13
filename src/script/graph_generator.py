#! /usr/bin/env python3
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import init_path
import constants as const


def argument_parser():
    parser = argparse.ArgumentParser(description="Graph for report")

    parser.add_argument('-g',
                        '--graph',
                        help=('Which graph to generate: ' +
                             '"imagenet", "segoverlap"'),
                        choices=['imagenet','segoverlap'],
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

def main():
    args = argument_parser()

    print(args)

    if args.graph == "imagenet":
        graph_imagenet()
    elif args.graph == "segoverlap":
        graph_segmentation_overlap()

if __name__ == "__main__":
    main()
