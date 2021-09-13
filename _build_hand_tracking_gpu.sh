#!/usr/bin/env bash

bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
   mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu

