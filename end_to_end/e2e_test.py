import cv2

import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import *
import end_to_end.dataparser_new
from end_to_end import dataparser_new


def test_video(video_path = '', model = None):
    vidcap = cv2.VideoCapture(video_path)
    frame_preds = []
    gesture_list = dataparser_new.get_gesture_list()
    if vidcap.isOpened():
        print("Reading video!")
        success, image = vidcap.read()
        while success:
            image = TF.to_tensor(image)
            with torch.no_grad():
                model.eval()
                output = model(image[None, ...].to(device))
                _, prediction = torch.max(output.data, dim=1)
                frame_preds.append(prediction)
            success, image = vidcap.read()
        vidcap.release()
    cv2.destroyAllWindows()

    return gesture_list[max(frame_preds, key=frame_preds.count)]

if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    net = models.resnet18(pretrained=True).to(device)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 6)
    net.fc = net.fc.to(device)

    net.load_state_dict(torch.load("../content/resnet18_e2e.pt"))

    print("********************* Model loaded!! **************************")
    print(test_video("../../e2e_test_videos/video_1.mp4", net))

