#!/usr/bin/python
# Facial detection model obtained from (and code adapted from): https://github.com/yeephycho/tensorflow-face-detection
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101
from pymediainfo import MediaInfo
import argparse
import sys
import time
import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import os

sys.path.append("..")

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

def anonymize_face_pixelate(image, blocks=3):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")

	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]

			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)

	# return the pixelated blurred image
	return image

def calculate_num_blocks_for_blur(x1, y1, x2, y2):
    return max(int(max(abs(y1-y2), abs(x1-x2))*BLOCK_SCALING_FACTOR), DEFAULT_NUM_BLOCKS)

def create_and_merge_blur_mask(frame, x1, y1, x2, y2):
    face = frame[y1:y2, x1:x2]
    # Apply a pixelated blur on the face
    face = anonymize_face_pixelate(face, calculate_num_blocks_for_blur(x1, y1, x2, y2))
    # Merge the blurred face to our final image
    frame[y1:y2, x1:x2] = face
    return frame

def process_video(input_file, output_path, score_threshold, enlarge_factor, skip_frames):
    print("Processing file: {} with confidence threshold {:3.2f}".format(input_file, score_threshold))

    split_name = os.path.splitext(os.path.basename(input_file))
    input_file_name = split_name[0]
    input_file_extension = split_name[1]
    output_file_name = os.path.join(output_path, '{}_blurred_auto_conf_{:3.2f}_skip{:d}{}'.format(input_file_name, 
        score_threshold, skip_frames, input_file_extension))

    # If output directory doesn't exist, create it
    if not os.path.isdir(output_path):
        print("Output directory doesn't exist, creating it now.")
        os.mkdir(output_path)

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    path_to_model = './model/frozen_inference_graph_face.pb'

    cap = cv2.VideoCapture(input_file)
    out = None
    # Specify codec to use when processing the video, eg. mp4v for a .mp4 file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Disable TensorFlow v2 deprecated warning compilation errors
    # TODO: (this code uses the v1 API and needs to be updated)
    tf.disable_v2_behavior()

    start_time = time.time()

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=detection_graph, config=config) as sess:
            count = 0
            last_bounding_boxes = list()
            while True:
                ret, frame = cap.read()
                if ret == 0:
                    break
                count += 1

                if count % 50 == 0:
                    print('Processing video {}, frame #{}...'.format(input_file_name, count))
                if out is None:
                    [h, w] = frame.shape[:2]
                    out = cv2.VideoWriter(output_file_name, fourcc, 25.0, (w, h))

                # Skip prediction on current frame if within skipping window and apply last calculated 
                # blurring masks (list of masks for each face)
                if count % (skip_frames + 1) != 0:
                    for i in range(len(last_bounding_boxes)):
                        bb = last_bounding_boxes[i]
                        frame = create_and_merge_blur_mask(frame, bb[0], bb[1], bb[2], bb[3])
                    out.write(frame)
                    continue
                else:
                    last_bounding_boxes.clear()

                image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # The array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                boxes, scores, classes = get_bounding_boxes(boxes, scores, classes, frame, score_thres=score_threshold,
                                                    enlarge_factor=enlarge_factor)
                for i in range(len(boxes)):
                    x1 = int(boxes[i][0])
                    y1 = int(boxes[i][1])
                    x2 = int(boxes[i][2])
                    y2 = int(boxes[i][3])
                    
                    last_bounding_boxes.append([x1, y1, x2, y2])
                    frame = create_and_merge_blur_mask(frame, x1, y1, x2, y2)

                out.write(frame)
            elapsed_time = time.time() - start_time
            # Get input video length in s
            media_info = MediaInfo.parse(input_file)
            input_length = media_info.tracks[0].duration/1000
            # Calculate processing ratio (s processing time/s video)
            proc_ratio = elapsed_time/input_length
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("Blurring video finished in {:.2f}s with processing ratio (s processing time/s video) {:2.4f} for video with dimensions {}x{}".format(elapsed_time, 
                proc_ratio, w, h))
            print("Output saved at {}".format(output_file_name))

            cap.release()
            out.release()

def main():
    args = get_arguments()
    input_path = args.input
    output_path = args.output
    score_thresholds = args.score_thresholds.split(",")
    enlarge_factor = args.enlarge_factor
    skip_frames = int(args.skip_frames)

    start_time = time.time()

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
                media_info = MediaInfo.parse(input_file)
                input_length += media_info.tracks[0].duration/1000
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