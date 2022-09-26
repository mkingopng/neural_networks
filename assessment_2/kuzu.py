"""
Michael Kingston

file_name: kuzu.py
Subject: ZZEN9444, CSE, UNSW

Question:

Notes:
    - how can i use logging?
    - overall accuracy =
    - Test set: Average loss: 0.3362, Accuracy: 9336/10000 (93%)
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        """

        """
        super(NetLin, self).__init__()  #
        self.linear = nn.Linear(28 * 28, 10)  #
        self.log_softmax = nn.LogSoftmax(dim=1)  #
        # self.lin_layer = nn.Linear(784, 10) # 28*28 = 784, 10 classifications

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x_1 = x.view(x.shape(0), -1)  # flattening the inputs.
        x_2 = self.log_softmax(self.linear(x_1))  #
        return x_2


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        """

        """
        super(NetFull, self).__init__()
        self.hid = nn.Linear(28 * 28, 200)  #
        self.tanh = nn.Tanh()  #
        self.out = nn.Linear(200, 10)  #
        self.log_softmax = nn.LogSoftmax(dim=1)  #

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x_1 = x.view(x.size(0), -1)  #
        x_2 = self.hid(x_1)  #
        x_3 = self.tanh(x_2)  #
        x_4 = self.out(x_3)  #
        x_5 = self.log_softmax(x_4)  #
        return x_5  #


class NetConv(nn.Module):
    """
    two convolutional layers and one fully connected layer,
    all using relu, followed by log_softmax
    :param:
    :return:
    """
    def __init__(self):
        super(NetConv, self).__init__()  #
        self.conv_1 = nn.Conv2d(1, 32, 6, padding=2)  #
        self.conv_2 = nn.Conv2d(32, 64, 6, padding=2)  #
        self.linear_1 = nn.Linear(43264, 64)  #
        self.linear_2 = nn.Linear(64, 10)  #
        self.relu = nn.ReLU()  #
        self.log_soft_max = nn.LogSoftmax()  #

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x_1 = self.relu(self.conv_2(self.relu(self.conv_1(x))))  #
        h = x_1.view(x_1.size(0), -1)  #
        output = self.log_soft_max(self.linear_2(self.relu(self.linear_1(h))))
        return output  #
