#!/usr/bin/env bash

bazel build --local_cpu_resources=2 -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
   mediapipe/examples/desktop/iris_tracking:iris_tracking_gpu

