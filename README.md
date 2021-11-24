# IIS_Project

This project consists of three phases and one optional phase: landmarks detection, gesture recognition, and the appropriate response of the gesture by Furhat Robot and with optional being End-to-End Learning. Currently, we implemented the landmarks detection of ASL Videos using ML/DL architectures.

This final project is a part of in [Intelligent Interactive Systems](https://www.uu.se/en/admissions/freestanding-courses/course/?kKod=1MD039&typ=1) course at Uppsala University.

# Directory Structure
Folder structure of videos and annotations dataset - Default ("../final_project_dataset/*")

Folder structure of imgs dataset - "../final_project_dataset_v0/imgs"

# Libraries, Methods & Utilities
Pytorch, pretrained Resnet18 model, MSE Loss, implemented on GPU - RTX-2060.

# Landmark Detection subsystem

Input - An anonymous video of ASL.

Expected Output -  X and Y coordinates of landmarks for every frame.

1. Convert all videos into images using function implemented in dataparser.py
2. Run train.py which includes preprocessing (to avoid overfitting/underfitting issues).

Good Luck!!