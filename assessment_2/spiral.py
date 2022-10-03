"""
spiral.py
ZZEN9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PolarNet(torch.nn.Module):
    def __init__(self, hidden_number):
        """
        first, the input (x, y) is converted to polar coordinates (r, a) with
            r = sqrt(x * x + y * y)
            a = atan(y, x)

        next, (r, a) is fed into a fully connected neural network with one
        hidden layer using tanh activation.

        finally a single output using sigmoid activation

        :param hidden_number:
        """
        super(PolarNet, self).__init__()
        self.linear_1 = nn.Linear(2, hidden_number)
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(hidden_number, 1)
        self.sigmoid = nn.Sigmoid()
        self.hidden_1 = 0

    def forward(self, input):
        """
        The conversion to polar coordinates should be included in your
        forward() method, so that the Module performs the entire task of
        conversion followed by network layers

        :param input:
        :return:
        """
        x = input[:, 0]
        y = input[:, 1]
        r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2)).view(-1, 1)
        a = torch.atan2(y, x).view(-1, 1)
        x_1 = torch.cat((r, a), 1)
        self.hidden_1 = self.tanh(self.linear_1(x_1))
        output = self.sigmoid(self.linear_2(self.hidden_1))
        return output


class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.linear_1 = nn.Linear(2, num_hid)
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(num_hid, num_hid)
        self.output = nn.Linear(num_hid, 1)
        self.sigmoid = nn.Sigmoid()
        self.hidden_1 = 0
        self.hidden_2 = 0

    def forward(self, input):
        """

        :param input:
        :return:
        """
        x = input[:, 0].view(-1, 1)
        y = input[:, 1].view(-1, 1)
        x_1 = torch.cat((x, y), 1)
        self.hidden_1 = self.tanh(self.linear_1(x_1))
        self.hidden_2 = self.tanh(self.linear_2(self.hidden_1))
        output = self.sigmoid(self.output(self.hidden_2))
        return output


def graph_hidden(net, layer, node):
    """
    suppress updating of gradients
    toggle batch norm, dropout
    to plot function computed by model
    :param net:
    :param layer:
    :param node:
    :return:
    """
    x_range = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    y_range = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    x_coord = x_range.repeat(y_range.size()[0])
    y_coord = torch.repeat_interleave(y_range, x_range.size()[0], dim=0)
    grid = torch.cat((x_coord.unsqueeze(1), y_coord.unsqueeze(1)), 1)
    with torch.no_grad():
        net.eval()
        net(grid)
        if layer == 1:
            pred = (net.hidden_1[:, node] >= 0.5).float()
        elif layer == 2:
            pred = (net.hidden_2[:, node] >= 0.5).float()
        plt.clf()
        plt.pcolormesh(x_range, y_range,
                       pred.cpu().view(y_range.size()[0], x_range.size()[0]),
                       cmap='Wistia')

