"""
kuzu.py
ZZEN9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetLin(nn.Module):
    """
    linear function followed by log_softmax
    input: 28 * 28 = 784
    output: 10
    """

    def __init__(self):
        super(NetLin, self).__init__()
        self.fully_connected = nn.Linear(784, 10)

    def forward(self, input):
        f_input = input.view(input.shape[0], -1)  # flatten inputs.
        output = F.log_softmax(self.fully_connected(f_input), dim=1)
        return output


class NetFull(nn.Module):
    """
    two fully connected tanh layers followed by log softmax
    input_1: features = 28 * 28 = 784
    hidden_layers: 150
    output_1: 200
    input_2: 200
    output_2: 10
    """

    def __init__(self):
        super(NetFull, self).__init__()
        hidden_layers = 150
        self.fc_layer_1 = nn.Linear(784, hidden_layers, bias=True)
        self.fc_layer_2 = nn.Linear(hidden_layers, 10, bias=True)

    def forward(self, input):
        """

        :param input:
        f_input: flattened input
        hidden:
        :return output:
        """
        f_input = input.view(input.shape[0], -1)
        hidden = torch.tanh(self.fc_layer_1(f_input))
        output = F.log_softmax(self.fc_layer_2(hidden), dim=1)
        return output


class NetConv(nn.Module):
    """
    two convolutional layers and one fully connected layer, all using relu,
    followed by log_softmax
    """

    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5,
                               padding=2)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=24, kernel_size=5,
                               padding=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_layer_1 = nn.Linear(1176, 159)
        self.fc_layer_2 = nn.Linear(159, 10)

    def forward(self, x):
        """
        flattening the inputs.
        :param input:
        :return output:
        """
        out = F.relu(self.conv1(x))
        out = self.max_pool(out)
        out = F.relu(self.conv2(out))
        out = self.max_pool(out)
        out = out.view(x.size(0), -1)
        out = F.relu(self.fc_layer_1(out))
        out = self.fc_layer_2(out)
        out = self.dropout(out)
        out = F.log_softmax(out, dim=1)
        return out
