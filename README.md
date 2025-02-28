# KNN Image Classifier

This repository contains an implementation of a **k-Nearest Neighbors (KNN)** classifier for image classification tasks. KNN is a simple yet effective machine learning algorithm used to classify data points based on feature similarity.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Features
- Implementation of the KNN algorithm for image-based datasets.
- Includes utility functions for loading and preprocessing datasets.
- Configurable parameters for KNN (e.g., number of neighbors).
- Works with custom datasets, including the provided face dataset.

## Requirements
- Python 3.x
- Required libraries: `numpy`, `scikit-learn`, `opencv-python`

## Usage
- Prepare the dataset or use the provided one in the [Face_Dataset/](https://github.com/mehdisn/knn-image-classifier/tree/main/Face_Dataset) folder.
- Run the classifier using:
`python main.py`
- Modify the main.py or knn.py script to adjust the KNN parameters or dataset path.

## Project Structure
- main.py: Entry point to run the KNN classifier.
- knn.py: Core implementation of the KNN algorithm.
- data_loader.py: Handles loading and preprocessing of images.
- utils.py: Contains helper functions used across the project.
- Face_Dataset/: Example dataset for testing the classifier.