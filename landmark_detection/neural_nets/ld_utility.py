import sys
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_rmse(loader, model, loss_fn, device):
    model.eval()
    num_examples = 0
    losses = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(device=device)
            target = target.to(device=device)

            scores = model(data)
            loss = loss_fn(scores[target != -1], target[target != -1])
            num_examples += scores[target != -1].shape[0]

            losses.append(loss.item())

    print(f"Loss on val: {(sum(losses)/num_examples) ** 0.5}")

def train_one_epoch(loader, model, optimizer, loss_fn, device, scaler=None):
    losses = []
    lp = tqdm(loader)
    num_examples = 0

    model.train()
    for batch_idx, (data, targets) in enumerate(lp):
        data = data.to(device = device)
        targets = targets.to(device = device)

        scores = model(data)
        scores[targets != -1] = -1
        loss = loss_fn(scores, targets)
        num_examples += torch.numel(scores[targets != -1])
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss average over epoch: {(sum(losses)/num_examples) ** 0.5}")