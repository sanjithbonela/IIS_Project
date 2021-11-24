"""TODO!"""

import time
import torch
import cv2
from landmark_detection.network_model import Network
import matplotlib.pyplot as plt


if __name__ == '__main__':
    start_time = time.time()

    images = cv2.imread("../../final_project_dataset_v0/test_imgs/test_set/test_img.jpg", 0)
    images = torch.from_numpy(images)
    print(images.shape)

    with torch.no_grad():
        best_network = Network()
        best_network.cuda()
        best_network.load_state_dict(torch.load('./content/landmarks_pretrained.pth'))
        best_network.eval()

        predictions = (best_network(images).cpu() + 0.5)
        predictions = predictions.view(-1, 20, 2)

        plt.figure(figsize=(10, 40))

        for img_num in range(8):
            plt.subplot(8, 1, img_num + 1)
            imgplot = plt.imshow(images[img_num].cpu().numpy().transpose(1, 2, 0).squeeze(), cmap='gray')
            plt.scatter(predictions[img_num, :, 0], predictions[img_num, :, 1], c='r', s=5)
            plt.show()

    print('Total number of test images: {}'.format(1))

    end_time = time.time()
    print("Elapsed Time : {}".format(end_time - start_time))