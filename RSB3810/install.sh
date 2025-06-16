#!/bin/bash
set -e

# Update package lists and install system dependencies
echo "Updating package lists and installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    v4l-utils \
    libv4l-dev \
    libjpeg-dev \
    libgstreamer1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    ffmpeg \
    python3-pip \
    python3-opencv \
    wget \
    unzip

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install numpy==1.24.4 tflite-runtime

# Download and set up the TensorFlow Lite model
echo "Downloading and preparing the TensorFlow Lite model..."
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
mv detect.tflite model.tflite
mv labelmap.txt labels.txt

echo "Setup complete. You can now run the demo with 'python3 demo.py'" 