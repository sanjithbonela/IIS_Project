import cv2
import glob
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import *
import end_to_end.dataparser_new
from end_to_end import dataparser_new
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def get_conf_matrix(path = '../../final_project_dataset_v2', net = None):
    list_directories = os.listdir(path)
    y_pred = []
    y_true = []
    score_count = 0
    total_count = 0

    if net is None:
        net = models.resnet18().to(device)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 6)
        net.fc = net.fc.to(device)

        net.load_state_dict(torch.load("../content/resnet18_e2e.pt"))

        print("********************* Model loaded!! **************************")

    for l in list_directories:
        if l.find('imgs') == -1:
            for video_file in glob.glob(path + "/" + l + "/videos/*.mp4"):
                total_count += 1
                pred_item = test_video(video_file, net).cpu().item()
                y_pred.append(pred_item)
                y_true.append(dataparser_new.get_mapping()[l])
                if y_true[len(y_true)-1] == pred_item:
                    score_count += 1

    print("Accuracy obtained: ", (score_count/total_count)*100)
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()


def real_time_video_e2e(threshold = 0.7, net = None):

    if net is None:
        net = models.resnet18().to(device)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 6)
        net.fc = net.fc.to(device)

        net.load_state_dict(torch.load("../content/resnet18_e2e_transform_v1.pt"))

        print("********************* Model loaded!! **************************")

    vidcap = cv2.VideoCapture(0)
    gesture_list = dataparser_new.get_gesture_list()

    if vidcap.isOpened():
        print("Started video!")
        success, image = vidcap.read()
        while success:
            image = cv2.resize(image, (480, 640))
            image = TF.to_tensor(image)
            image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            with torch.no_grad():
                net.eval()
                output = net(image[None, ...].to(device))
                output.data = torch.exp(output.data)/torch.sum(torch.exp(output.data), dim=1)
                res, prediction = torch.max(output.data, dim=1)
                if res.data > threshold:
                    print(gesture_list[prediction])
                else:
                    print("No gesture detected!")
            success, image = vidcap.read()
            c = cv2.waitKey(1)
            if c == 27:
                break
        vidcap.release()
    cv2.destroyAllWindows()

    print("Done live streaming!")


def test_video(video_path = '', net = None):
    if net is None:
        net = models.resnet18().to(device)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 6)
        net.fc = net.fc.to(device)

        net.load_state_dict(torch.load("../content/resnet18_e2e_transform_v1.pt"))

        print("********************* Model loaded!! **************************")
    vidcap = cv2.VideoCapture(video_path)
    frame_preds = []
    gesture_list = dataparser_new.get_gesture_list()

    if vidcap.isOpened():
        print("Reading video!")
        success, image = vidcap.read()
        while success:
            image = TF.to_tensor(image)
            # image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            with torch.no_grad():
                net.eval()
                output = net(image[None, ...].to(device))
                _, prediction = torch.max(output.data, dim=1)
                frame_preds.append(prediction)
            success, image = vidcap.read()
        vidcap.release()
    cv2.destroyAllWindows()

    print('Done prediction!')
    # Different return statements to enable different features.
    # return max(frame_preds, key=frame_preds.count)
    # return {video_path : gesture_list[max(frame_preds, key=frame_preds.count)]}
    return {"gesture_name": gesture_list[max(frame_preds, key=frame_preds.count)]}

def pred_multiple_videos(path = ''):
    net = models.resnet18().to(device)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 6)
    net.fc = net.fc.to(device)

    net.load_state_dict(torch.load("../content/resnet18_e2e_transform_v1.pt"))

    print("********************* Model loaded!! **************************")

    pred_list = []
    for video in glob.glob(path):
        pred_list.append(test_video(video, net))
    return pred_list


if __name__ == '__main__':

    # pred_list = pred_multiple_videos('../../final_project_dataset_v2/ASL_letter_U/videos/*.mp4')
    get_conf_matrix()
    # pred = test_video('../../final_project_dataset_v1/ASL_letter_A/videos/video_1.mp4')
    #
    # print(pred.cpu().item())
    # print(dataparser_new.get_mapping()['ASL_letter_A'])
    # real_time_video_e2e()

    # for e in pred_list:
    #     print(e)

