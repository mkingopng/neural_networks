#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn
import re
from config import device



def tokenise(sample):
    processed = sample.split(" ")
    return processed


def preprocessing(sample):
    new_list = []
    cop = re.compile("[^a-zA-Z\s\d]")
    for i in sample:
        i = cop.sub(' ', i)
        if (len(i) > 1):
            new_list.append(i)
    return new_list
    # return sample


def postprocessing(batch, vocab):
    return batch


stopWords = ["a", "about", "above", "after", "again", "against", "ain", "all",
             "am", "an", "and", "any", "are",
             "as", "at", "be", "because", "been", "before", "being", "below",
             "between", "both", "but", "by",
             "can", "d", "did", "do", "does", "doing", "don", "down",
             "during", "each", "few", "for", "from", "further", "had", "hadn",
             "has",
             "have", "having", "he", "her", "here", "hers", "herself", "him",
             "himself", "his", "how", "i",
             "if", "in", "into", "is",
             "it", "it's", "its", "itself", "just", "ll", "m",
             "ma", "me", "mightn", "more", "most", "my", "myself", "needn",
             "now", "o", "of",
             "off", "on", "once", "only", "or", "other", "our", "ours",
             "ourselves", "out",
             "over", "own", "re", "s", "same", "shan", "she", "she's",
             "should", "should've", "so", "some",
             "such", "t", "than", "that",
             "that'll", "the", "their", "theirs", "them", "themselves", "then",
             "there", "these",
             "they", "this", "those", "through", "to", "too", "under", "until",
             "up", "ve", "very",
             "was", "we", "were", "what", "when", "where",
             "which", "while", "who", "whom", "why", "will", "with",
             "y", "you", "you'd", "you'll", "you're", "you've", "your",
             "yours",
             "yourself", "yourselves", "could", "he'd", "he'll", "he's",
             "here's", "how's",
             "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll",
             "that's", "there's",
             "they'd", "they'll", "they're", "they've", "we'd", "we'll",
             "we're", "we've", "what's",
             "when's", "where's", "who's", "why's", "would"]

word_vector_switch = '6B100'
if word_vector_switch == '6B50':
    wordVectors = GloVe(name='6B', dim=50)
elif word_vector_switch == '6B100':
    wordVectors = GloVe(name='6B', dim=100)


def convertNetOutput(ratingOutput, categoryOutput):
    ratingOutput = torch.argmax(ratingOutput, dim=-1)
    categoryOutput = torch.argmax(categoryOutput, dim=-1)
    return ratingOutput, categoryOutput


class network(tnn.Module):
    def __init__(self):
        super(network, self).__init__()
        input_size = 100
        hidden_size = 256
        is_bidirection = False
        lstm_layers = 1
        self.gru_layers = 2
        self.rating_encoder = tnn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bidirectional=False,
            batch_first=True,
            dropout=0.5,
        )
        self.business_category_encoder = tnn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bidirectional=False,
            batch_first=True,
            dropout=0.4,
        )
        self.business_category_GRU = tnn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.gru_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0,
        )
        self.rating_fc1 = tnn.Linear(hidden_size, 50)
        self.rating_fc2 = tnn.Linear(128, 50)
        self.business_category_fc1 = tnn.Linear(hidden_size, 50)
        self.business_category_fc2 = tnn.Linear(128, 50)
        self.relu = tnn.ReLU()
        self.decoder1 = tnn.Linear(50, 2)
        self.decoder2 = tnn.Linear(50, 5)

    def forward(self, input, length):
        input = tnn.utils.rnn.pack_padded_sequence(input, length,
                                                   batch_first=True)

        # rating related forward part
        rating_outputs, (rating_hidden, rating_cells) = self.rating_encoder(
            input)
        rating_hidden = self.rating_fc1(rating_hidden[0, :, :])
        # rating_hidden = self.rating_fc2(rating_hidden)
        x = self.decoder1(rating_hidden)

        # business category related forward part
        business_category_outputs, business_category_hidden = self.business_category_GRU(
            input)
        if self.gru_layers == 1:
            business_category_hidden = self.business_category_fc1(
                business_category_hidden[0])
        elif self.gru_layers == 2:
            business_category_hidden = self.business_category_fc1(
                business_category_hidden[0])  # resize
        y = self.decoder2(business_category_hidden)

        return (x, y)


'''
class network(tnn.Module):
    def __init__(self):
        super(network, self).__init__()
        input_size = 100
        hidden_size = 256
        is_bidirection = False
        lstm_layers = 1
        self.gru_layers = 2
        self.encoder = tnn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = lstm_layers,
            bidirectional = False,
            batch_first = True,
            dropout = 0.5,
        )
        self.fc1 = tnn.Linear(hidden_size,128)
        self.decoder = tnn.Linear(128,10)

    def forward(self, input, length):
        input = tnn.utils.rnn.pack_padded_sequence(input,length,batch_first = True)

        outputs,(hidden,cell) = self.encoder(input)
        hidden = self.fc1(hidden[0])
        output = self.decoder(hidden)

        return(0,output)
'''


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.rating_loss = tnn.CrossEntropyLoss()
        self.category_loss = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget,
                categoryTarget):

        ratingloss = self.rating_loss(ratingOutput, ratingTarget)
        categoryloss = self.category_loss(categoryOutput, categoryTarget)
        return ratingloss + categoryloss


'''
def convert_target(ratingTarget,categoryTarget):
    for k in ratingTarget:
'''

net = network()
lossFunc = loss()
# lossFunc = tnn.MSELoss()


trainValSplit = 0.8
batchSize = 32
epochs = 8
optimiser = toptim.SGD(net.parameters(), lr=0.05, momentum=0.9)
# optimiser = toptim.Adam(net.parameters(),lr= 0.1)
