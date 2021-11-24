import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from LandmarksDataset import FaceLandmarksDataset
from Transforms import Transforms
from network_model import Network
from network_model import CustomNet
import matplotlib.pyplot as plt


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

    dataset = FaceLandmarksDataset(Transforms())
    len_valid_set = int(0.1 * len(dataset))
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

    torch.autograd.set_detect_anomaly(True)
    network = Network()
    try:
        network.load_state_dict(torch.load("./content/landmarks_pretrained.pth"))
        print("Model loaded Successfully!")
    except:
        print("Model not found! Initiating from scratch")
        network = Network(isPretrained=True)
    network = network.to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    loss_min = np.inf

    start_time = time.time()
    print("Training started!")
    for epoch in range(1, num_epochs + 1):

        loss_train = 0
        loss_valid = 0
        running_loss = 0

        network.train()
        for step in range(1, len(train_loader) + 1):
            images, landmarks = next(iter(train_loader))

            images = images.to(device)
            landmarks = landmarks.view(landmarks.size(0), -1).to(device)
            #print(images.shape)
            predictions = network(images)

            # clear all the gradients before calculating them
            optimizer.zero_grad()

            # find the loss for the current step
            loss_train_step = criterion(predictions, landmarks)

            # calculate the gradients
            loss_train_step.backward()

            # update the parameters
            optimizer.step()

            loss_train += loss_train_step.item()
            running_loss = loss_train / step

            print_overwrite(step, len(train_loader), running_loss, 'train')

        network.eval()
        with torch.no_grad():

            for step in range(1, len(valid_loader) + 1):
                images, landmarks = next(iter(valid_loader))

                images = images.to(device)
                landmarks = landmarks.view(landmarks.size(0), -1).to(device)

                predictions = network(images)

                # find the loss for the current step
                loss_valid_step = criterion(predictions, landmarks)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid / step

                print_overwrite(step, len(valid_loader), running_loss, 'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        train_loss_list.append(loss_train)
        valid_loss_list.append(loss_valid)
        epoch_list.append(epoch-1)

        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
        print('--------------------------------------------------')

        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(network.state_dict(), './content/landmarks_pretrained_cont.pth')
            print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
            print('Model Saved\n')

    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time() - start_time))
    plt.plot(epoch_list, train_loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Average training loss over an epoch")
    plt.show()


if __name__ == '__main__':
    train_model(learning_rate=0.0001)