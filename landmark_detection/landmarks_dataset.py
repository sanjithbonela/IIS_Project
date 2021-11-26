import cv2
import os
import numpy as np
from torch.utils.data import Dataset
import dataparser


class FaceLandmarksDataset(Dataset):

    def __init__(self, transform=None):

        self.image_filenames = []
        self.landmarks = []
        # self.crops = []
        self.transform = transform
        self.root_dir = '../../final_project_dataset_v0/imgs'
        lst = os.listdir(self.root_dir)
        pd_df = dataparser.concatenated_df()

        for filename in lst:
            self.image_filenames.append(os.path.join(self.root_dir, filename))

            landmark = self.getFilteredLandmarks(pd_df, filename)
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks)

        landmarks = landmarks - 0.5

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
            x_coord = int(row["y"])
            y_coord = int(row["x"])
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
