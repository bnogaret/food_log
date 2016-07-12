#! /usr/bin/env python3
import argparse

from google.protobuf import text_format

import caffe
from caffe.proto import caffe_pb2
from caffe.draw import draw_net_to_file

import init_path

import constants as const


def argument_parser():
    parser = argparse.ArgumentParser(description="Generate image to visualize the CNN")
    
    parser.add_argument('-m',
                        '--model',
                        help='Which CNN to visualize: ' + \
                             '"seg": the one used for segmentation ' + \
                             '"des": the one used for description',
                        choices=['seg', 'des'],
                        default='seg',
                        type=str)
    
    parser.add_argument('-p',
                        '--phase',
                        help='Which network phase to draw: ' + \
                              '"test": test phase ' + \
                              '"train": train phase ' + \
                              '"all": all layers',
                        choices=['test', 'train', 'all'],
                        default="all",
                        type=str)
    
    args = parser.parse_args()
    return args


def save_cnn_graph(path_model, name_model, phase, phase_name):
    net_parameter = caffe_pb2.NetParameter()
    text_format.Merge(open(path_model).read(), net_parameter)
    draw_net_to_file(net_parameter, const.PATH_TO_IMAGE_DIR + "/network_graph_" + name_model + "_" + phase_name + ".png", "BT", phase)


def main():
    args = argument_parser()
    
    print(args)
    
    arg_model = {
        'seg'   : const.PATH_TO_SEG_MODEL_DEF,
        'des'   : const.PATH_TO_DESCRI_MODEL_DEF
    }
    
    arg_phase = {
        'train' : caffe.TRAIN,
        'test'  : caffe.TEST,
        'all'   : None
    }
    
    save_cnn_graph(arg_model[args.model], args.model, arg_phase[args.phase], args.phase)

if __name__ == "__main__":
    main()

