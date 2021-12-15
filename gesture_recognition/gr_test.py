import cv2
import glob
import pandas as pd
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import *
from kafka import KafkaConsumer
import json
import kafka_consumer
from gesture_recognition import dataparser_gr

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def getLandmarks(pd_df, frame_number=None, video_id=None):
    filtered_df = pd_df[(pd_df.frame == frame_number) & (pd_df.video_idx == video_id) & (pd_df.joint != "hand_position")]
    landmark = []
    for index, row in filtered_df.iterrows():
        x_coord = int(row["y"])/480
        y_coord = int(row["x"])/640
        landmark.append([x_coord, y_coord])
    return landmark

def test_landmarks(landmarks, net = None):
    with torch.no_grad():
        net.eval()
        output = net(landmarks).to(device)
        _, prediction = torch.max(output.data, dim=1)
    return dataparser_gr.get_gesture_list()[prediction]

def pred_multiple_videos(path = ''):
    net = models.resnet18().to(device)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 6)
    net.fc = net.fc.to(device)

    net.load_state_dict(torch.load("./content/resnet18_e2e.pt"))

    print("********************* Model loaded!! **************************")

    pred_list = []
    for video in glob.glob(path):
        pred_list.append(test_video(video, net))
    return pred_list

def function:
    
    consumer = KafkaConsumer('my_topic',group_id='my-group_1',bootstrap_servers=['localhost:9092'])
    consumer.message ==

    pd_df = pd.read_csv("../../final_project_dataset_v1/ASL_letter_B/annotations.csv", index_col=None, header=0)
    landmarks = getLandmarks(pd_df, frame_number=0, video_id=1)
    landmarks = np.array(landmarks).astype('float32')
    landmarks = landmarks.reshape(-1, 40)
    landmarks = landmarks-0.5
    landmarks = torch.from_numpy(landmarks).to(device)

    layers_shape = [40, 200, 400, 600, 800, 6]

    net = torch.nn.Sequential(
        torch.nn.Linear(layers_shape[0], layers_shape[1]),
        torch.nn.ReLU(),
        torch.nn.Linear(layers_shape[1], layers_shape[2]),
        torch.nn.ReLU(),
        torch.nn.Linear(layers_shape[2], layers_shape[3]),
        torch.nn.ReLU(),
        torch.nn.Linear(layers_shape[3], layers_shape[4]),
        torch.nn.ReLU(),
        torch.nn.Linear(layers_shape[4], layers_shape[5]),
    ).to(device)

    net.load_state_dict(torch.load("../content/gr_ffn_normalized.pt"))

    print("********************* Model loaded!! **************************")

    #pred_list = pred_multiple_videos('../../final_project_dataset_v1/ASL_letter_C/videos/*.mp4')
    gesture_nm = test_landmarks(landmarks, net)
#Insert here the producer for kafka
    return gesture_nm

if __name__ == '__main__':
    function()


