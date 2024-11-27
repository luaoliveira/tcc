# Neural Networks to segment forest areas in drone images | October, 2024 Version.

## Description

The purpose of this project is to provide a script to identify and segment forest regions in drone images. 

## Project overview

The inputs for the models are yml configuration files that can be filled with parameters and hyperparameters.

The outputs from the models are: 

- Segmented binary masks highlighting the forest/non forest areas.
- Overlaying masks for the binary masks over original images
- Segmentation metrics : IoU, F1-Score, Precision, Recall.
- Confusion Matrix images
- Training error curves
- Number of pixels (estimated area of forest/non forest )

## Repository structure



