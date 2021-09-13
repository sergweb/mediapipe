#!/usr/bin/env bash

echo Starting Drowsiness Detector ... 

curdir=$(dirname "$(realpath $0)")

echo "Home directory: ${curdir}"

cd ${curdir}  

# remove logs and video older than 7 days
find . -type f -name "drowsiness_detector.*.log" -mtime +7 -exec rm -f {} \;
find . -type f -name "drowsiness_detector_gps.*.csv" -mtime +7 -exec rm -f {} \;
find . -type f -name "drowsiness_detector_video.*.avi" -mtime +7 -exec rm -f {} \; 

DISPLAY=:0.0 GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_gpu \
--calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_tracking_gpu_serg.pbtxt \
--use_gps=yes \
--gps_speed_threshold=10 \
--gps_logger=yes \
--gps_logger_file=drowsiness_detector_gps.%T.csv \
--gps_logger_interval=3 \
--gps_audio=yes \
--display_video=yes \
--flip_video=yes \
--output_video=yes \
--output_video_path=drowsiness_detector_video.%T.avi \
--output_video_duration=300 \
--closed_eyes_aspect_ratio=0.33 \
--drowsiness_interval=3 \
--max_fps=30 \
--alarm_interval=8 \
--volume=0.40 \
2>&1 | tee drowsiness_detector.$(date +"%Y%m%d_%H%M%S").log
