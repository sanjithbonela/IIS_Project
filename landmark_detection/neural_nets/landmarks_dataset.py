import cv2
import os
import imageio
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from landmark_detection.neural_nets import dataparser


class LandmarksDataset(Dataset):

    def __init__(self, transform=None):

        self.image_filenames = []
        self.landmarks = []
        # self.crops = []
        self.transform = transform
        self.root_dir = '../../../final_project_dataset_v0/imgs'
        lst = os.listdir(self.root_dir)
        pd_df = dataparser.concatenated_df()

        for filename in lst:
            self.image_filenames.append(os.path.join(self.root_dir, filename))

            landmark = self.getFilteredLandmarks(pd_df, filename)
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')
        self.landmarks = self.landmarks.reshape(-1, 40)

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = imageio.imread(self.image_filenames[index])
        landmarks = self.landmarks[index]
        # print(landmarks.shape)
        # image = TF.to_tensor(image)
        # image = TF.normalize(image, [0.5], [0.5])

        if self.transform:
            image, landmarks = self.transform(image, landmarks)

        #landmarks = landmarks - 0.5

        return image, landmarks

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
            if row["y"] == 0 and row["x"] == 0:
                x_coord = -1
                y_coord = -1
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
