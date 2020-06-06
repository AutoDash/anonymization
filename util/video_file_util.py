#!/usr/bin/python
# Helper functions for validating and working with video files

# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import os

def validate_input_files(input_path, accepted_file_extensions, score_thresholds, output_path, enlarge_factor, skip_frames):
    pass
    # TODO: finish this
    # # Validate input if it's a single file
    # if os.path.isfile(input_path):
    #     if input_path.endswith(accepted_file_extensions):
    #         for threshold in score_thresholds:
    #             process_video(input_path, output_path, float(threshold), enlarge_factor, skip_frames)
    #         exit(0)
    #     else:
    #         print('Not a valid file.')
    #         exit(1)

    # # Process each file in the input directory
    # if not(os.path.isdir(input_path)) or not(os.listdir(input_path)):
    #     print('No input files found or invalid directory specified.')
    #     exit(1)

