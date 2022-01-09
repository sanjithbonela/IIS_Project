# IIS_Project

This project consists of three phases and one optional phase: landmarks detection, gesture recognition, and the appropriate response of the gesture by Furhat Robot and with optional being End-to-End Learning. Currently, we implemented the landmarks detection of ASL Videos using ML/DL architectures.

This final project is a part of in [Intelligent Interactive Systems](https://www.uu.se/en/admissions/freestanding-courses/course/?kKod=1MD039&typ=1) course at Uppsala University.

# Directory Structure
Folder structure of videos and annotations dataset - Default ("../final_project_dataset/*")

Folder structure of imgs dataset - "../final_project_dataset_v0/imgs"

# Libraries, Methods & Utilities
Pytorch, Mediapipe, pretrained Resnet18 model, implemented on GPU - RTX-2060.

# Communication between subsystems - Kafka
Kafka is an open-source messaging framework which contains producer and consumer. The producer send the message to the server and it stores in topic, and consumer receives the message from the server.
 
Check these links for installation - [Documentation](https://kafka.apache.org/intro), [Installation](https://kafka.apache.org/downloads), [Setup and startup instructions](https://kafka.apache.org/quickstart).

Create 3 topics for communication between first and second, second and third and end-to-end and third subsystems.

Make sure those names of the topics match the ones given in code. 

# Landmark Detection subsystem

Install all dependencies through requirements.txt

Input - An anonymous video of ASL.

Expected Output -  X and Y coordinates of landmarks for every frame.

#. Method I
1. Convert all videos into images using function implemented in dataparser.py
2. Run train.py which includes preprocessing (to avoid overfitting/underfitting issues).

However, this model isn't performing upto expectation and so another method was implemented.

### Evaluation 
    Euclidean distance between predicted and groundtruth landmarks of a video

# Method II
Using open source module, mediapipe, the landmarks were detected and the performance was decent enough.

# End to End Learning subsystem

Input - An anonymous video of ASL.

Expected output - One of the alphabet in [A, B, C, L, R, U]

1. Pretrained Resnet18 model was used and the trained model was stored in content folder.
2. Run e2e_test.py to test the model for both saved and real time video.

### Evaluation
    Accuracy measure, confusion matrices

Good Luck!!