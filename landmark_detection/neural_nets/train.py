import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from landmark_detection.neural_nets.landmarks_dataset import LandmarksDataset
from landmark_detection.neural_nets.transforms import Transforms
from landmark_detection.neural_nets.network_model import Network, CustomNet
import matplotlib.pyplot as plt
from landmark_detection.neural_nets import ld_utility
from landmark_detection.neural_nets.ld_utility import train_one_epoch


def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))

    sys.stdout.flush()

def train_model(learning_rate, num_epochs = 10):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    dataset = LandmarksDataset(Transforms())
    len_valid_set = int(0.3 * len(dataset))
    len_train_set = len(dataset) - len_valid_set

    train_loss_list = []
    valid_loss_list = []
    epoch_list = []

    print("The length of Train set is {}".format(len_train_set))
    print("The length of Valid set is {}".format(len_valid_set))

    train_dataset, valid_dataset, = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])

    # shuffle and batch the datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True)
    images, landmarks = next(iter(train_loader))
    print(images.shape)
    print(landmarks.shape)

    torch.autograd.set_detect_anomaly(True)
    network = Network()
    try:
        network.load_state_dict(torch.load("../../content/landmarks_r18.pth"))
        print("*************** Model loaded Successfully! ****************")
    except:
        print("*************** Model not found! Initiating pretrained version ******************")
        network = Network(isPretrained=True)
    network = network.to(device)

    criterion = nn.MSELoss(reduction="sum").to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    print("Training started!")

    for epoch in range(1, num_epochs + 1):
        ld_utility.get_rmse(valid_loader, network, criterion, device)
        ld_utility.train_one_epoch(train_loader, network, optimizer, criterion, device, scaler)
        torch.save(network.state_dict(), '../../content/landmarks_r18.pth')
        print('Model Saved\n')

    end_time = time.time()
    print(f"Time taken for training: {end_time - start_time}")

if __name__ == '__main__':
    train_model(learning_rate=1e-3)