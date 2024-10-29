import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU()
        )
        self.linear3 = nn.Linear(in_features=84, out_features=10)


        # self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=0, stride=1)
        # self.s2 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        # self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1)
        # self.s4 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        # self.c5 = nn.Linear(in_features=16*5*5,out_features=120)
        # self.f6 = nn.Linear(in_features=120,out_features=84)
        # self.output = nn.Linear(in_features=84,out_features=10)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0),-1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = F.softmax(x)
        return x

