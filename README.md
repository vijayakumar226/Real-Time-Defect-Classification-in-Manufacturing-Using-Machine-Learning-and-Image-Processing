# Real-Time Defect Classification in Manufacturing Using Machine Learning and Image Processing

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-brightgreen)](https://scikit-learn.org/)

## Abstract / Overview

This repository implements an automated quality control system for injection-molded circuit breakers. It combines classical image processing techniques with machine learning classification to detect and categorize manufacturing defects in real time. The pipeline extracts texture, color, and geometric features from images and evaluates multiple classifiers to identify defective and non-defective parts.

## Key Features

- Image preprocessing with grayscale conversion, histogram equalization, Gaussian blur, and Canny edge detection.
- Texture analysis using GLCM (Gray Level Co-occurrence Matrix) and LBP (Local Binary Patterns).
- Color and geometric feature extraction for robust defect discrimination.
- Multi-class defect classification including Short Fill, Crack, Scrolling, Shrink, Flash Fill, and Non-Defect.
- Comparison of machine learning models: Random Forest, Support Vector Machine, and K-Nearest Neighbors.

## Technology Stack

- Python
- OpenCV for image processing
- Scikit-Learn for machine learning
- NumPy and pandas for data handling
- Matplotlib / Seaborn for visualization (optional)

## Dataset Note

> The dataset used in this project is an **in-house industrial dataset** and is kept strictly confidential. The dataset is **NOT included** in this repository. Users must provide their own image dataset to run the code.

## Installation and Setup Instructions

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd "web application"
   ```
2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Prepare your own image dataset with labeled defect classes.

## Usage

1. Place your image files in a local folder structure or update the dataset loader in the code to point to your images.
2. Update any dataset path variables in `app.py` or other scripts to reference your data source.
3. Run the main script:
   ```bash
   python app.py
   ```
4. Monitor output logs and predictions for defect classification results.

> Note: Since the repository does not contain the proprietary dataset, you must supply image data for Short Fill, Crack, Scrolling, Shrink, Flash Fill, and Non-Defect classes.

## Model Performance & Results

The best performing model in this project was the Random Forest classifier.

| Model | Accuracy | F1-Score |
|------:|:--------:|:--------:|
| Random Forest | 94.1% | 0.94 |
| Support Vector Machine | 92.4% | - |
| K-Nearest Neighbors | 90.2% | - |

## Citation / Academic Publication

This work was published at ICVADV-2026.

- Paper: https://ieeexplore.ieee.org/document/11470348

If you use this repository for research or development, please cite the ICVADV-2026 publication.
