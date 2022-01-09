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
from end_to_end.label_dataset_e2e import FaceGestureDataset
from end_to_end.e2e_transforms import E2E_Transforms
from end_to_end import dataparser_new


def imshow(inp, title=None):
    inp = inp.cpu() if device else inp
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # abc = plt.imshow(inp)
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.001)

def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()

if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    batch_size = 16
    learning_rate = 1e-3

    dataparser_new.convert_video_to_images(path='../../final_project_dataset_v1')

    # dataset = FaceGestureDataset(path='../../final_project_dataset_v1', transform=E2E_Transforms())
    dataset = FaceGestureDataset(path='../../final_project_dataset_v1')
    len_valid_set = int(0.2 * len(dataset))
    len_train_set = len(dataset) - len_valid_set

    train_loss_list = []
    valid_loss_list = []
    epoch_list = []

    print("The length of Train set is {}".format(len_train_set))
    print("The length of Valid set is {}".format(len_valid_set))

    train_dataset, valid_dataset, = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    images, labels = next(iter(train_dataloader))
    print("images-size:", images.shape)

    out = torchvision.utils.make_grid(images)
    print("out-size:", out.shape)
    # imshow(out, title="abcd")
    print('\n')

    net = models.resnet18(pretrained=True)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 6)
    net.fc = net.fc.to(device)

    n_epochs = 10
    print_every = 10
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_dataloader)
    for epoch in range(1, n_epochs + 1):
        epoch_list.append(epoch)
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
                torch.save(net.state_dict(), '../content/resnet18_e2e_transform_v1.pt')
                print('Improvement-Detected, save-model')

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(epoch_list, train_loss, label = 'train-loss')
    ax1.plot(epoch_list, val_loss, label = 'valid-loss')
    ax1.set(xlabel = 'Epochs', ylabel = 'Loss')
    ax1.set_title('Loss vs epoch')

    ax2.plot(epoch_list, train_acc, label='train-accuracy')
    ax2.plot(epoch_list, val_acc, label='valid-accuracy')
    ax2.set(xlabel='Epochs', ylabel='Accuracy')
    ax2.set_title('Accuracy vs epoch')
    plt.savefig('../../E2E_loss_accuracy.png')
    plt.show()
