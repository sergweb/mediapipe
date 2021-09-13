#!/usr/bin/env bash

gst-launch-1.0 filesrc location=alarm.wav ! wavparse ! autoaudiosink -e
