# @Author: Avinash Kadimisetty <avinash>
# @Date:   2018-12-27T12:32:08-06:00
# @Project: ConvolutionalNeuralNetwork
# @Filename: model_architecture.py
# @Last modified by:   avinash
# @Last modified time: 2018-12-27T12:52:50-06:00


# =============================================================================
# The CNN model architecture is defined hre
# =============================================================================


import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, stride=1, padding=2, kernel_size=4,
                      out_channels=64),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, stride=1, padding=2, kernel_size=4,
                      out_channels=64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Dropout2d(p=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, stride=1, padding=2, kernel_size=4,
                      out_channels=64),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, stride=1, padding=2, kernel_size=4,
                      out_channels=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, stride=1, padding=2, kernel_size=4,
                      out_channels=64),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, stride=1, padding=0, kernel_size=3,
                      out_channels=64),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=64, stride=1, padding=0, kernel_size=3,
                      out_channels=64),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=64, stride=1, padding=0, kernel_size=3,
                      out_channels=64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.5)
        )
        self.fc1 = nn.Linear(in_features=1024, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=K)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out
