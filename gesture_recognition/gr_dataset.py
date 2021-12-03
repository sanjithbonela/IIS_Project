import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from gesture_recognition import dataparser_gr
import torchvision.transforms.functional as TF


class GestureRecognitionDataset(Dataset):

    def __init__(self, path = '../../final_project_dataset_v0', pd_df=None):

        #self.image_filenames = []
        self.labels = []
        self.landmarks = []
        # self.crops = []
        self.root_dir = path + "/imgs"
        lst = os.listdir(self.root_dir)
        #pd_df = dataparser_e2e.concatenated_df(path = path)

        for filename in lst:
            #self.image_filenames.append(os.path.join(self.root_dir, filename))

            landmark = self.getFilteredLandmarks(pd_df, filename)
            self.landmarks.append(landmark)
            label = self.getFilteredLabels(filename)
            self.labels.append(label)

        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels).long()
        self.landmarks = np.array(self.landmarks).astype('float32')
        self.landmarks = self.landmarks.reshape(-1, 40)

        assert len(self.landmarks) == len(self.labels)

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, index):
        landmarks = self.landmarks[index]
        labels = self.labels[index]

        landmarks = landmarks - 0.5

        return landmarks, labels

    def getFilteredLabels(self, img_name):
        nms = img_name.split("$$")
        gesture_name = nms[0]
        gesture_map = dataparser_gr.get_mapping()
        # label = [0 for _ in range(0, 6)]
        # label[gesture_map[gesture_name]] = 1
        return gesture_map[gesture_name]

    def getFilteredLandmarks(self, pd_df, img_name):
        nms = img_name.split("$$")
        gesture_name = nms[0]
        video_id = int((nms[1].split("_"))[1])
        frame_num = int((nms[2].split("_"))[1])
        #video_id = int(nms[1].split("_")[2])
        filtered_df = pd_df[(pd_df.gesture == gesture_name) & (pd_df.frame == frame_num) & (pd_df.video_idx == video_id) & (pd_df.joint != "hand_position")]
        landmark = []
        for index, row in filtered_df.iterrows():
            x_coord = int(row["y"])/480
            y_coord = int(row["x"])/640
            landmark.append([x_coord, y_coord])
        return landmark

'''
if __name__ == '__main__':
    dataset = FaceLandmarksDataset(Transforms())
    image, landmarks = dataset[0]
    landmarks = (landmarks + 0.5)
    plt.figure(figsize=(10, 10))
    imgplot = plt.imshow(image.numpy().squeeze(), cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=8)
    plt.show()
    '''
