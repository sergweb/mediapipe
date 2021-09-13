#!/usr/bin/env bash

bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
   mediapipe/examples/desktop/upper_body_pose_tracking:upper_body_pose_tracking_gpu

