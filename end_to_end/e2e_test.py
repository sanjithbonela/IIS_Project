import cv2
import glob
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import *
import end_to_end.dataparser_new
from end_to_end import dataparser_new

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def test_video(video_path = '', net = None):
    if net is None:
        net = models.resnet18().to(device)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 6)
        net.fc = net.fc.to(device)

        net.load_state_dict(torch.load("./content/resnet18_e2e.pt"))

        print("********************* Model loaded!! **************************")
    vidcap = cv2.VideoCapture(video_path)
    frame_preds = []
    gesture_list = dataparser_new.get_gesture_list()

    if vidcap.isOpened():
        print("Reading video!")
        success, image = vidcap.read()
        while success:
            image = TF.to_tensor(image)
            with torch.no_grad():
                net.eval()
                output = net(image[None, ...].to(device))
                _, prediction = torch.max(output.data, dim=1)
                frame_preds.append(prediction)
            success, image = vidcap.read()
        vidcap.release()
    cv2.destroyAllWindows()

    print('Done prediction!')
    return {"gesture_name": gesture_list[max(frame_preds, key=frame_preds.count)]}

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


# if __name__ == '__main__':
#
#     pred_list = pred_multiple_videos('../../final_project_dataset_v1/ASL_letter_C/videos/*.mp4')
#
#     print(pred_list)

