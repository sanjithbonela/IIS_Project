import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import copy
import os
from gesture_recognition.gr_dataset import GestureRecognitionDataset
from gesture_recognition import dataparser_gr


def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()

if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    batch_size = 128
    learning_rate = 0.001

    concat_pd_df = dataparser_gr.concatenated_df()
    print("Loaded concatenated df!")
    dataset = GestureRecognitionDataset(path='../../final_project_dataset_v0', pd_df=concat_pd_df)
    len_valid_set = int(0.3 * len(dataset))
    len_train_set = len(dataset) - len_valid_set

    train_loss_list = []
    valid_loss_list = []
    epoch_list = []

    print("The length of Train set is {}".format(len_train_set))
    print("The length of Valid set is {}".format(len_valid_set))

    train_dataset, valid_dataset, = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    landmarks, labels = next(iter(train_dataloader))
    print("landmarks-size :", landmarks.shape)
    print("labels-size: ", labels.shape)

    # out = torchvision.utils.make_grid(images)
    # print("out-size:", out.shape)
    # imshow(out, title="abcd")

    layers_shape = [40, 200, 400, 600, 800, 6]

    net = torch.nn.Sequential(
        torch.nn.Linear(layers_shape[0], layers_shape[1]),
        torch.nn.ReLU(),
        torch.nn.Linear(layers_shape[1], layers_shape[2]),
        torch.nn.ReLU(),
        torch.nn.Linear(layers_shape[2], layers_shape[3]),
        torch.nn.ReLU(),
        torch.nn.Linear(layers_shape[3], layers_shape[4]),
        torch.nn.ReLU(),
        torch.nn.Linear(layers_shape[4], layers_shape[5]),
    )

    # net = models.resnet18(pretrained=True)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

    # num_ftrs = net.fc.in_features
    # net.fc = nn.Linear(num_ftrs, 6)
    # net.fc = net.fc.to(device)

    n_epochs = 30
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_dataloader)
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()
            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            # print(pred)
            # print(target_)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)
            if (batch_idx) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')
        batch_loss = 0
        total_t = 0
        correct_t = 0
        with torch.no_grad():
            net.eval()
            for data_t, target_t in (valid_dataloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t.data, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss / len(valid_dataloader))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
            if network_learned:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), '../content/gr_ffn_normalized.pt')
                print('Improvement-Detected, save-model')