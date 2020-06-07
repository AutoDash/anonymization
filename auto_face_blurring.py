#!/usr/bin/python
# This module automatically applies a blurring mask to faces from a given input set of videos using a
# MobileNet SSD an efficient convolutional neural network based model, for facial detection.
#
# Facial detection model obtained from (and code adapted from):
# https://github.com/yeephycho/tensorflow-face-detection
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101
import argparse
import time
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

from threading import Thread

import util.video_file_util as video_file_util
import util.detection_util as detection_util

__version__ = '1.0'

class AutoFaceBlurrer:
    DEFAULT_INPUT_PATH = './dataset/input/'
    DEFAULT_OUTPUT_PATH = './dataset/output/'
    ACCEPTED_FILE_EXTENSION = ".mp4"

    DEFAULT_SCORE_THRESH = "0.15"
    DEFAULT_ENLARGE_FACTOR = 0.1
    DEFAULT_SKIP_FRAMES = 0

    DEFAULT_NUM_BLOCKS = 3
    BLOCK_SCALING_FACTOR = 0.03

    DEFAULT_BUFFER_SIZE = 512

    def __init__(self, input, output, score_thresholds, enlarge_factor, skip_frames, frame_buffer_size):
        self.input_path = input
        self.output_path = output
        self.score_thresholds = score_thresholds.split(",")
        self.enlarge_factor = enlarge_factor
        self.skip_frames = int(skip_frames)
        self.frame_buffer_size = int(frame_buffer_size)

    def run(self):
        start_time = time.time()

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        path_to_model = './model/frozen_inference_graph_face.pb'

        detection_graph = detection_util.load_detection_graph(path_to_model)
        detection_util.set_memory_growth(detection_graph)

        # TODO: In TensorFlow 2, Sessions have been deprecated in favor of tf.Functions.
        # However, with the existing implementation, we'd have to exclusively use Tensor objects
        # for computation. Currently the benefits of this conversion do not seem to be significant,
        # so this is left for future work.
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Validate input if it's a single file
            if os.path.isfile(self.input_path):
                if self.input_path.endswith(self.ACCEPTED_FILE_EXTENSION):
                    for threshold in self.score_thresholds:
                        video_file_util.process_video(self.input_path, self.output_path,
                            sess, self.skip_frames, self.BLOCK_SCALING_FACTOR, self.DEFAULT_NUM_BLOCKS, 
                                detection_graph, float(threshold), self.enlarge_factor, self.frame_buffer_size,
                                    'image_tensor:0', 'detection_boxes:0', 'detection_scores:0', 
                                        'detection_classes:0')
                    exit(0)
                else:
                    print('Not a valid file.')
                    exit(1)

            # Process each file in the input directory
            if not(os.path.isdir(self.input_path)) or not(os.listdir(self.input_path)):
                print('No input files found or invalid directory specified.')
                exit(1)

            files_processed = 0
            files_produced = 0

            # Get total input video length in s
            input_length = 0

            for file in os.listdir(self.input_path):
                if file.endswith(self.ACCEPTED_FILE_EXTENSION):
                    for threshold in self.score_thresholds:
                        input_file = os.path.join(self.input_path, file)
                        # Get total input video length in s
                        input_length += video_file_util.get_video_length(input_file)
                        # Process the video
                        video_file_util.process_video(input_file, self.output_path,
                            sess, self.skip_frames, self.BLOCK_SCALING_FACTOR, self.DEFAULT_NUM_BLOCKS,
                                detection_graph, float(threshold), self.enlarge_factor, self.frame_buffer_size,
                                    'image_tensor:0', 'detection_boxes:0', 
                                        'detection_scores:0', 'detection_classes:0')
                        files_produced += 1
                    files_processed +=1


            elapsed_time = time.time() - start_time
            proc_ratio = elapsed_time/input_length
            
            print("{} files were processed and {} were produced in {:.4f}s with average processing ratio "
                "(s processing time/s video) {:2.4f}.".format(files_processed, files_produced, elapsed_time, proc_ratio))

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
        parser.add_argument('--input', default=AutoFaceBlurrer.DEFAULT_INPUT_PATH,
                        help="The relative path to the input video dataset to be processed. Can be a directory or a single file. "
                            "Default: {}".format(AutoFaceBlurrer.DEFAULT_INPUT_PATH))
        parser.add_argument('--output', default=AutoFaceBlurrer.DEFAULT_OUTPUT_PATH,
                        help="The relative path to the output directory to be used for storing the processed videos. "
                            "Default: {}".format(AutoFaceBlurrer.DEFAULT_OUTPUT_PATH))
        parser.add_argument("--score-thresholds", default=AutoFaceBlurrer.DEFAULT_SCORE_THRESH,
                        help="A comma separated list of confidence score thresholds (eg. 0.05,0.10,0.15). For every input video, "
                                "a variant output file for each threshold is provided. The threshold for each variant is used to "
                                "determine which bounding boxes from the model prediction is accepted (every bounding box that has "
                                "a confidence score higher than this threshold). "
                                "Default: {}".format(AutoFaceBlurrer.DEFAULT_SCORE_THRESH))
        parser.add_argument("--enlarge-factor", type=float, default=AutoFaceBlurrer.DEFAULT_ENLARGE_FACTOR,
                            help="The factor by which the detected facial bounding boxes are enlarged before "
                                "applying the blurring mask. "
                                "Default: {}".format(AutoFaceBlurrer.DEFAULT_ENLARGE_FACTOR))
        parser.add_argument("--skip-frames", type=float, default=AutoFaceBlurrer.DEFAULT_SKIP_FRAMES,
                            help="The number of frames to skip after performing the ML prediction before predicting again. "
                                "This will help speed up the video processing. For every frame that is skipped, the blurring mask for the last calculated "
                                "bounding boxes will remain. "
                                "Default: {}".format(AutoFaceBlurrer.DEFAULT_SKIP_FRAMES))
        parser.add_argument("--frame-buffer-size", type=float, default=AutoFaceBlurrer.DEFAULT_SKIP_FRAMES,
                    help="The number of frames to hold in the FIFO buffer which helps optimizes the processing"
                        "of videos in the program. A higher number requires more memory. "
                        "Default: {}".format(AutoFaceBlurrer.DEFAULT_BUFFER_SIZE))

        args = parser.parse_args()

        return args

def main():
    args = get_arguments()
    blurrer = AutoFaceBlurrer(args.input, args.output, 
        args.score_thresholds, args.enlarge_factor, args.skip_frames, args.frame_buffer_size)
    blurrer.run()


if __name__ == '__main__':
    main()