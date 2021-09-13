#!/usr/bin/env bash

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_gpu   --calculator_graph_config_file=mediapipe/graphs/face_mesh/face_mesh_desktop_live_gpu.pbtxt
