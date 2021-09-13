#!/usr/bin/env bash

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/upper_body_pose_tracking/upper_body_pose_tracking_gpu \
	--calculator_graph_config_file=mediapipe/graphs/pose_tracking/upper_body_pose_tracking_gpu.pbtxt

#--output_video_path=output.mp4

