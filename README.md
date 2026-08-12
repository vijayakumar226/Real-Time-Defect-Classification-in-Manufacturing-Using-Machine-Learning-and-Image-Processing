# Real-Time Defect Classification in Manufacturing Using Deep Learning and Image Processing

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Object_Detection-orange)](https://github.com/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen)](https://opencv.org/)

## Abstract / Overview

This repository implements an automated quality control system for injection-molded circuit breakers using deep learning and object detection. The pipeline utilizes a YOLO (You Only Look Once) architecture to identify, localize, and classify surface defects in real-time, outputting bounding boxes and confidence scores suitable for high-speed industrial manufacturing environments.

## Workflow & Key Features

- **Data Annotation:** The custom dataset was manually annotated using `labelImg` to create precise bounding boxes for object detection training.
- **Real-Time Detection:** Employs a trained YOLO model to perform high-speed inference.
- **Multi-Class Localization:** Detects and localizes multiple defect categories including Short Fill, Crack, Scrolling, Shrink, Flash Fill, and Non-Defect.
- **Visual Output:** Generates bounding box overlays with class labels and confidence scores (e.g., `broken 0.97`, `scrolling 0.97`).

## Technology Stack

- Python
- YOLO (You Only Look Once)
- PyTorch
- OpenCV for image processing and bounding box visualization
- labelImg for dataset annotation

## Dataset Note

> The dataset used in this project is an **in-house industrial dataset** and is kept strictly confidential. The dataset is **NOT included** in this repository. Users must provide their own image dataset, annotated in YOLO format, to run the training or inference code.

## Installation and Setup Instructions

1. Clone this repository:
   ```bash
   git clone <https://github.com/vijayakumar226/Real-Time-Defect-Classification-in-Manufacturing-Using-Machine-Learning-and-Image-Processing.git>
   cd "web application"y for research or development, please cite the ICVADV-2026 publication( https://ieeexplore.ieee.org/document/11470348 ).
