# Neural Networks to segment forest areas in drone images | 2024 V.

## Description

The purpose of this project is to develop a script to identify and segment forest regions in drone images, providing an estimation of forest area in them.

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

- :file_folder: masks 
- :file_folder: my_models
  - :file_folder: configs -> configuration yml files to be filled before running the models
  - :file_folder: training images -> images used for training / fine tuning
  - :file_folder: training masks -> binary masks used for training 
  - :file_folder: validation images -> images used for validation/test 
  - :file_folder: validation masks -> images used for validation/test (during training masks located here won't be used for backpropagation)
