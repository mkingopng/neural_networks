#!/usr/bin/env python3
"""
_student.py

UNSW ZZEN9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
a3main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as
a basic tokenise function.  You are encouraged to modify these to improve
the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""
import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import os
import random
import re
import sys
import time
import unicodedata
import numpy
import string
import mlflow
import nltk
import spacy
import pandas
import scipy
import sklearn
from config import device

"""
parameters and hyperparameters
    nb: the optmizer is at the end
"""
dimensions = 300  # 50, 200
wordVectors = GloVe(name='6B', dim=dimensions)
trainValSplit = 0.8  # # fix_me
batchSize = 128  # 32, 128 # fix_me
epochs = 10  # # fix_me
lr = 0.002  # 0.02, 0.01, 0.001, 0.002, 0.005,
hidden_size = 300  # 50, 100, 200, 256 # fix_me
dropout = 0.3  # 0.5 # fix_me
num_layers = 2  # 1, # fix_me

stopWords = {
    "a", "about", "above", "after", "again", "against", "ain", "all", "am",
    "an", "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "d", "did", "do",
    "does", "doing", "don", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn", "has", "have", "having", "he", "her", "here",
    "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me",
    "mightn", "more", "most", "my", "myself", "needn", "now", "o",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "re", "s", "same", "shan",
    "she", "she's", "should", "should've", "so", "some", "such", "t",
    "than", "that", "that'll", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "ve", "very",
    "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "y", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's",
    "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll",
    "that's", "there's", "they'd", "they'll", "they're", "they've",
    "we'd", "we'll", "we're", "we've", "what's", "when's", "where's",
    "who's", "why's", "would"
}


def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    processed = sample.split()
    return processed


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    new_list = []
    CLASSES = {'0': "zero ", '1': "one", '2': "two", '3': "three",
               '4': "four", '5': "five"}
    for i in sample:
        # mask stopwords
        if i not in stopWords:
            # convert numbers to words
            if i in CLASSES.keys():
                new_list.append(CLASSES[i])
            else:
                # remove punctuations
                new_list.append(i.strip(string.punctuation))
    for i in sample:
        i = re.compile("[^a-zA-Z\s\d]").sub(' ', i)  #
        if len(i) > 1:
            new_list.append(i)
    return new_list


# MAX_WORDS = 50000
def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    return batch


"""
The following determines the processing of label data (ratings)
"""


def convertNetOutput(rating_output, category_output):
    rating_output = torch.argmax(rating_output, dim=-1)
    category_output = torch.argmax(category_output, dim=-1)
    return rating_output.to(device), category_output.to(device)


"""
The following determines the model
"""


class network(tnn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(
            input_size=dimensions,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,  # try True
            batch_first=True,  # try False
            dropout=dropout
        )
        self.linear_f_rating1 = tnn.Linear(hidden_size, 128)  # 200, 64
        self.linf2 = tnn.Linear(128, 2)  # 64, 1
        self.lin_fcat1 = tnn.Linear(hidden_size, 128)  #
        self.lin_fcat2 = tnn.Linear(128, 5)  # 128, 6
        self.relu = tnn.ReLU()

    def forward(self, input, length):
        embeded = tnn.utils.rnn.pack_padded_sequence(
            input,
            length.cpu(),
            batch_first=True
        )
        output, (hidden, cell) = self.lstm(embeded)
        out_rat = self.linf2(self.relu(self.linear_f_rating1(hidden[1])))
        out_cat = self.lin_fcat2(self.relu(self.lin_fcat1(hidden[1])))
        return out_rat, out_cat


class loss(tnn.Module):
    """
    Class for creating the loss function. The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss = tnn.CrossEntropyLoss()

    def forward(self, rating_output, category_output, rating_target,
                category_target):
        rate_loss = self.loss(rating_output, rating_target)
        category_loss = self.loss(category_output, category_target)
        return rate_loss + category_loss


net = network()
lossFunc = loss()

"""
The following determines training options 
"""
# optimiser = toptim.SGD(net.parameters(), lr=lr)
optimiser = toptim.Adam(net.parameters(), lr=lr)
# optimiser = toptim.AdamW(net.parameters(), lr=lr)
