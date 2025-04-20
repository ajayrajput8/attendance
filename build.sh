#!/usr/bin/env bash

# Install OpenCV dependencies
apt-get update && apt-get install -y \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  ffmpeg \
  libopencv-dev
