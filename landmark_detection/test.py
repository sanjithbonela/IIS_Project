"""TODO!"""

import time
import torch
import cv2

from torch.utils.data import Dataset
from landmark_detection.landmarks_dataset import FaceLandmarksDataset
from landmark_detection.network_model import Network
from landmark_detection.transforms import Transforms
import matplotlib.pyplot as plt


if __name__ == '__main__':

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    start_time = time.time()

    dataset = FaceLandmarksDataset(Transforms())
    len_test_set = int(1 * len(dataset))
    len_vd_set = len(dataset) - len_test_set
    test_dataset, vd_dataset = torch.utils.data.random_split(dataset, [len_test_set, len_vd_set])

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    with torch.no_grad():
        best_network = Network()
        best_network = best_network.to(device)
        best_network.load_state_dict(torch.load('../content/landmarks_pretrained_cont_2.pth'))
        best_network.eval()

        images, landmarks = next(iter(test_loader))
        images = images.to(device)
        predictions = (best_network(images).cpu() + 0.5)
        predictions = predictions.view(-1, 20, 2).cpu()

        plt.figure()

        for img_num in range(6):
            plt.subplot(6, 1, img_num + 1)
            img = images[img_num].cpu().numpy().transpose(1, 2, 0).squeeze()
            for l in predictions[img_num, :][:]:
                img[int(l[1].item()), int(l[0].item())] = cv2.COLOR_YUV420P2RGBA
            #img[int(predictions[img_num, :][:][0]), int(predictions[img_num, :][:][1])] = [0, 0, 255]
            cv2.imshow("final", img)
            cv2.waitKey(0)
            imgplot = plt.imshow(img, cmap='gray')
            plt.scatter(predictions[img_num, :, 0], predictions[img_num, :, 1], c='r', s=5)
            plt.show()

    print('Total number of test images: {}'.format(1))

    end_time = time.time()
    print("Elapsed Time : {}".format(end_time - start_time))