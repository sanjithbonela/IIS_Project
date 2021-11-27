import torch
import torch.nn as nn
from torchvision import models


class Network(nn.Module):
    def __init__(self, num_classes=40, isPretrained=False):
        super().__init__()
        self.model_name = 'resnet50'
        self.model = models.resnet50(pretrained=isPretrained)
        #self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class CustomNet(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.trans_conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=1, stride=1)
        self.convL = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)
        self.act4 = nn.ReLU()
        self.fffc = nn.Linear(in_features=16*614400, out_features=num_classes)
        self.skip = nn.Identity()

    def forward(self, x):

        conv_out1 = self.conv1(x)
        act_out1 = self.act1(conv_out1)
        maxpool_out1 = self.maxpool1(act_out1)

        conv_out2 = self.conv2(maxpool_out1)
        act_out2 = self.act2(conv_out2)
        maxpool_out2 = self.maxpool2(act_out2)

        conv_out3 = self.conv3(maxpool_out2)
        act_out3 = self.act3(conv_out3)

        trans_out1 = self.trans_conv1(act_out3 + self.skip(conv_out3))
        trans_out2 = self.trans_conv2(trans_out1+self.skip(conv_out2))
        trans_out3 = self.trans_conv3(trans_out2 + self.skip(conv_out1))

        conv_lst = self.convL(trans_out3)
        act_outL = self.act4(conv_lst)
        ftn = torch.flatten(act_outL, 1)
        linear = self.fffc(ftn, 40)

        return linear