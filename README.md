# Car-Plate-Detection-YOLO

Car Plate Recognition using YOLO for automatic license plate detection, cropping, and visualization in images with Python, OpenCV, and the Ultralytics YOLO library. A deep learning-based computer vision project that performs license plate recognition.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Architecture and Results](#model-architecture-and-results)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Demo Video](#demo-video)
- [Future Enhancements](#future-enhancements)

## Introduction

This project provides a robust solution for automatic license plate detection and recognition using deep learning. Built with YOLOv11, this computer vision system can accurately identify and locate license plates in both images and videos. The entire project is powered by Python, OpenCV, and the Ultralytics YOLO library, with an interactive web application created using Streamlit.

## Features

- **Real-time Detection**: Utilizes the high-speed YOLOv11 model for efficient object detection.
- **Image & Video Support**: Processes both static images and video streams for license plate detection.
- **Interactive Web App**: A user-friendly Streamlit application for easy interaction.
- **Customizable Thresholds**: Users can adjust the confidence threshold to fine-tune detection sensitivity.
- **Clean Visualization**: Displays the original and processed images/videos with bounding boxes around detected license plates.

## Dataset

The model was trained on a large, public dataset to ensure high accuracy.

- **Dataset Name**: License Plate Recognition
- **Source**: [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)
- **Total Images**: 10,125

## Technologies Used

The project's technology stack is designed for performance and ease of use.

- **Ultralytics YOLO**: The state-of-the-art YOLOv11 model for real-time object detection.
- **Python**: The core programming language.
- **OpenCV**: Essential for image and video manipulation.
- **Streamlit**: For building a beautiful, interactive, and easy-to-use web application.
- **Roboflow**: Platform for dataset management and download.

## Model Architecture and Results

The detection model is built on the advanced YOLOv11 architecture, optimized for performance and precision.

### Training Results

| Epoch  | GPU Memory | Box Loss | Class Loss | DFL Loss | Instances | Size |
|--------|------------|----------|------------|----------|-----------|------|
| 15/15  | 5.46G      | 0.9805   | 0.3915     | 1.112    | 23        | 640  |

### Validation Metrics

The model demonstrates excellent performance on the validation set, achieving high scores across key metrics.

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|--------|-----------|-----------|--------|--------|-----------|
| All   | 2046   | 2132      | 0.976     | 0.961  | 0.983  | 0.701     |

## Getting Started

### Prerequisites

- **Python**: A working Python environment (version 3.11 recommended). Download from the [official Python website](https://www.python.org/).
- **GPU**: Optional but recommended for faster model inference (e.g., T4 GPU on Google Colab).

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/amr10w/Car-Plate-Detection-YOLO.git
   cd Car-Plate-Detection-YOLO
   ```

2. Install the necessary libraries using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` contains:

   ```text
    streamlit
    ultralytics
    opencv-python
    numpy
    pillow
   ```

### Running the Application

1. Launch the Streamlit application from your terminal:

   ```bash
   streamlit run app/app.py
   ```

2. Your web browser will automatically open, allowing you to upload images or videos for license plate detection.

## Demo Video

https://github.com/user-attachments/assets/7c66b242-d53a-4de8-82c1-e4af0e1b5edb

## Future Enhancements

- **Optical Character Recognition (OCR)**: Add a second stage to read characters on detected license plates.
- **Performance Optimization**: Explore model quantization or other techniques to improve inference speed for real-time video processing.
- **Multiple Model Support**: Allow users to select different YOLO models (e.g., small, medium, large) within the Streamlit app.
