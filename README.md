# IIS_Project

In this part of the project, we implement the landmarks detection of ASL Videos using ML/DL architectures. The project consists of three phases and one optional phase: landmarks detection, gesture recognition, and the appropriate response of the gesture by Furhat Robot and with optional being End-to-End Learning.

This the first part of final project is in Intelligent Interactive Systems course at Uppsala University.

Input - An anonymous video of ASL.

Expected Output -  X and Y coordinates of landmarks for every frame.

# Directory Structure
Folder structure of videos and annotations dataset - Default ("../final_project_dataset/*")

Folder structure of imgs dataset - "../final_project_dataset_v0/imgs"

# Libraries & Utilities
Pytorch, pretrained Resnet18 model, MSE Loss, implemented on GPU - RTX-2060.

# Usage
1. Convert all videos into images using function implemented in dataparser.py
2. Run train.py which includes preprocessing (to avoid overfitting/underfitting issues).

Good Luck!!