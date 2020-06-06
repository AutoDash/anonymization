#!/usr/bin/python
# This module automatically applies a blurring mask to faces from a given input set of videos using a MobileNet SSD
# an efficient convolutional neural network based model, for facial detection.
#
# Facial detection model obtained from (and code adapted from): https://github.com/yeephycho/tensorflow-face-detection
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101
import argparse
import time
import numpy as np
import tensorflow as tf
import os

import util.video_file_util as video_file_util
import util.detection_util as detection_util

DEFAULT_INPUT_PATH = './dataset/input/'
DEFAULT_OUTPUT_PATH = './dataset/output/'
ACCEPTED_FILE_EXTENSION = ".mp4"

DEFAULT_SCORE_THRESH = "0.15"
DEFAULT_ENLARGE_FACTOR = 0.1
DEFAULT_SKIP_FRAMES = 0

DEFAULT_NUM_BLOCKS = 3
BLOCK_SCALING_FACTOR = 0.03

__version__ = '1.0'

def get_arguments():
    """
    Parse the arguments
    :return:
    """
    parser = argparse.ArgumentParser(
        prog='Blur faces in video using openCV',
        description="This module automatically applies a blurring mask to faces from a given input set of videos using a MobileNet SSD, " 
        "an efficient convolutional neural network based model, for facial detection.",
        epilog="Developed by: AutoDash")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(__version__))
    parser.add_argument('--input', default=DEFAULT_INPUT_PATH,
                    help="The relative path to the input video dataset to be processed. Can be a directory or a single file. "
                          "Default: {}".format(DEFAULT_INPUT_PATH))
    parser.add_argument('--output', default=DEFAULT_OUTPUT_PATH,
                    help="The relative path to the output directory to be used for storing the processed videos. "
                         "Default: {}".format(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--score-thresholds", default=DEFAULT_SCORE_THRESH,
                    help="A comma separated list of confidence score thresholds (eg. 0.05,0.10,0.15). For every input video, "
                            "a variant output file for each threshold is provided. The threshold for each variant is used to "
                            "determine which bounding boxes from the model prediction is accepted (every bounding box that has "
                            "a confidence score higher than this threshold). "
                            "Default: {}".format(DEFAULT_SCORE_THRESH))
    parser.add_argument("--enlarge-factor", type=float, default=DEFAULT_ENLARGE_FACTOR,
                        help="The factor by which the detected facial bounding boxes are enlarged before "
                             "applying the blurring mask. "
                             "Default: {}".format(DEFAULT_ENLARGE_FACTOR))
    parser.add_argument("--skip-frames", type=float, default=DEFAULT_SKIP_FRAMES,
                        help="The number of frames to skip after performing the ML prediction before predicting again. "
                             "This will help speed up the video processing. For every frame that is skipped, the blurring mask for the last calculated "
                             "bounding boxes will remain. "
                             "Default: {}".format(DEFAULT_SKIP_FRAMES))

    args = parser.parse_args()

    return args

def process_video(input_file, output_path, score_threshold, enlarge_factor, skip_frames):
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    path_to_model = './model/frozen_inference_graph_face.pb'

    detection_graph = detection_util.load_detection_graph(path_to_model)
    detection_util.set_memory_growth(detection_graph)

    # TODO: In TensorFlow 2, Sessions have been deprecated in favor of tf.Functions.
    # However, with the existing implementation, we'd have to exclusively use Tensor objects
    # for computation. Currently the benefits of this conversion do not seem to be significant,
    # so this is left for future work.
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        print("Processing file: {} with confidence threshold {:3.2f}".format(input_file, score_threshold))
        split_name = os.path.splitext(os.path.basename(input_file))
        input_file_name = split_name[0]
        input_file_extension = split_name[1]
        output_file_name = os.path.join(output_path, '{}_blurred_auto_conf_{:3.2f}_skip{:d}{}'.format(input_file_name, 
            score_threshold, skip_frames, input_file_extension))

        start_time = time.time()

        # If output directory doesn't exist, create it
        if not os.path.isdir(output_path):
            print("Output directory doesn't exist, creating it now.")
            os.mkdir(output_path)

        (width, height, video_length) = video_file_util.process_video(input_file, input_file_name, output_file_name,
            sess, skip_frames, BLOCK_SCALING_FACTOR, DEFAULT_NUM_BLOCKS, detection_graph, score_threshold,
                enlarge_factor, 'image_tensor:0', 'detection_boxes:0', 'detection_scores:0', 'detection_classes:0')

        elapsed_time = time.time() - start_time

        proc_ratio = video_file_util.calculate_proc_ratio(elapsed_time, video_length)

        print("Blurring video finished in {:.2f}s with processing ratio (s processing time/s video) {:2.4f} for video with dimensions {}x{}".format(elapsed_time, 
            proc_ratio, width, height))
        print("Output saved at {}".format(output_file_name))

def main():
    args = get_arguments()
    input_path = args.input
    output_path = args.output
    score_thresholds = args.score_thresholds.split(",")
    enlarge_factor = args.enlarge_factor
    skip_frames = int(args.skip_frames)

    start_time = time.time()

    # Validate input if it's a single file
    if os.path.isfile(input_path):
        if input_path.endswith(ACCEPTED_FILE_EXTENSION):
            for threshold in score_thresholds:
                process_video(input_path, output_path, float(threshold), enlarge_factor, skip_frames)
            exit(0)
        else:
            print('Not a valid file.')
            exit(1)

    # Process each file in the input directory
    if not(os.path.isdir(input_path)) or not(os.listdir(input_path)):
        print('No input files found or invalid directory specified.')
        exit(1)

    files_processed = 0
    files_produced = 0

    # Get total input video length in s
    input_length = 0

    for file in os.listdir(input_path):
        if file.endswith(ACCEPTED_FILE_EXTENSION):
            for threshold in score_thresholds:
                input_file = os.path.join(input_path, file)
                # Get total input video length in s
                input_length += video_file_util.get_video_length(input_file)
                # Process the video
                process_video(input_file, output_path, float(threshold), enlarge_factor, skip_frames)
                files_produced += 1
            files_processed +=1


    elapsed_time = time.time() - start_time
    proc_ratio = elapsed_time/input_length
    
    print("{} files were processed and {} were produced in {:.4f}s with average processing ratio "
        "(s processing time/s video) {:2.4f}.".format(files_processed, files_produced, elapsed_time, proc_ratio))

if __name__ == '__main__':
    main()