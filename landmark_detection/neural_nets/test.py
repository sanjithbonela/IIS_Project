"""TODO!"""

import time
import torch
import cv2
import imageio
import numpy as np
# from PIL import Image

from torch.utils.data import Dataset
from landmark_detection.neural_nets.network_model import Network
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


if __name__ == '__main__':

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    start_time = time.time()

    img_path = "D:\\Stuff\\Semester 3\\Period_2\\Intelligent Int Systems\\Project\\final_project_dataset_v1\\test_imgs\\ASL_letter_A$$video_0$$frame_1$$.jpg"
    image_org = imageio.imread(img_path)
    image = image_org
    # image = Image.fromarray(image)
    image = np.array(image)/255.0
    image = np.swapaxes(image, 1, 2)
    image = np.swapaxes(image, 0, 1)
    image = torch.from_numpy(image).float()
    # image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    best_network = Network()
    best_network = best_network.to(device)
    best_network.load_state_dict(torch.load('../../content/landmarks_new_imp.pth'))

    with torch.no_grad():
        best_network.eval()
        image = image.to(device)
        predictions = (best_network(image[None, ...]).cpu())
        predictions = predictions.view(-1, 20, 2) + 0.5
        # print(predictions)

        plt.figure()

        for img_num in range(1):
            plt.subplot(1, 1, img_num + 1)
            img = image_org#.cpu().numpy().transpose(1, 2, 0).squeeze()
            for l in predictions[img_num, :][:]:
                img[int(l[1].item()), int(l[0].item())] = cv2.COLOR_YUV420P2RGBA
            #img[int(predictions[img_num, :][:][0]), int(predictions[img_num, :][:][1])] = [0, 0, 255]
            cv2.imshow("final", img)
            cv2.waitKey(0)
            imgplot = plt.imshow(img, cmap='gray')
            plt.scatter(predictions[img_num, :, 0] * 480, predictions[img_num, :, 1] * 640, c='r', s=5)
            plt.show()

    print('Total number of test images: {}'.format(1))

    end_time = time.time()
    print("Elapsed Time : {}".format(end_time - start_time))