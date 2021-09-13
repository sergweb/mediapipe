#!/usr/bin/env bash

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_gpu \
	--calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live_gpu.pbtxt

#--output_video_path=output.mp4

