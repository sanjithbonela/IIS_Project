import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from end_to_end import dataparser_new
import torchvision.transforms.functional as TF


class FaceGestureDataset(Dataset):

    def __init__(self, path = '../../final_project_dataset_v0', transform=None):

        self.image_filenames = []
        self.labels = []
        # self.crops = []
        self.transform = transform
        self.root_dir = path + "/imgs"
        lst = os.listdir(self.root_dir)
        #pd_df = dataparser_e2e.concatenated_df(path = path)

        for filename in lst:
            self.image_filenames.append(os.path.join(self.root_dir, filename))

            label = self.getFilteredLabels(filename)
            self.labels.append(label)

        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels).long()

        assert len(self.image_filenames) == len(self.labels)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index])
        image = cv2.resize(image, (480, 640))
        # cv2.imshow("abcd", image)
        labels = self.labels[index]
        # image = TF.to_tensor(image)
        # image = TF.normalize(image, [0.5], [0.5])
        # labels = torch.from_numpy(labels).long()
        if self.transform:
            image = self.transform(image)

        #landmarks = landmarks - 0.5

        return image, labels

    def getFilteredLabels(self, img_name):
        nms = img_name.split("$$")
        gesture_name = nms[0]
        gesture_map = dataparser_new.get_mapping()
        # label = [0 for _ in range(0, 6)]
        # label[gesture_map[gesture_name]] = 1
        return gesture_map[gesture_name]


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
