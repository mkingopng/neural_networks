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
lr = 0.005  # 0.02, 0.01, 0.001, 0.002, 0.005,
hidden_size = 300  # 50, 100, 200 # fix_me
dropout = 0.3  # fix_me
num_layers = 2  # fix_me

stopWords = {
    "a", "about", "above", "after", "again", "against", "ain", "all", "am",
    "an", "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "d", "did", "do",
    "does", "doing", "don", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "ll", "m", "ma", "me",
    "mightn", "more", "most", "my", "myself", "needn", "now", "o", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "re", "s", "same", "shan", "she", "she's", "should",
    "shouldve", "so", "some", "such", "t", "than", "that", "that'll", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "ve",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "y", "you", "youd", "youll",
    "youre", "youve", "your", "yours", "yourself", "yourselves", "could",
    "hed", "hell", "hes", "heres", "hows", "id", "ill", "im", "ive",
    "lets", "ought", "shed", "shell", "thats", "theres", "theyd",
    "theyll", "theyre", "theyve", "wed", "well", "were", "weve",
    "whats", "whens", "wheres", "whos", "whys", "would"
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
    NUMBER_CONSTANT = {'0': "zero ", '1': "one", '2': "two", '3': "three",
                       '4': "four", '5': "five"}
    for i in sample:
        # filter some stopwords
        if i not in stopWords:
            # convert numbers to words
            if i in NUMBER_CONSTANT.keys():
                new_list.append(NUMBER_CONSTANT[i])
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
    # batch = [[x if x<MAX_WORDS else 0 for x in example]
    #       for example in batch]
    return batch


"""
The following determines the processing of label data (ratings)
"""


def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    ratingOutput = torch.argmax(ratingOutput, dim=-1)
    categoryOutput = torch.argmax(categoryOutput, dim=-1)
    return ratingOutput.to(device), categoryOutput.to(device)


"""
The following determines the model
"""


class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """
    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(
            input_size=dimensions,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout
        )
        self.linear_f_rating1 = tnn.Linear(hidden_size, 128)  # 200, 64
        self.linear_f_rating2 = tnn.Linear(128, 2)  # 64, 1
        self.linear_f_category_1 = tnn.Linear(hidden_size, 128)  #
        self.linear_f_category_2 = tnn.Linear(128, 5)  #
        self.relu = tnn.ReLU()

    def forward(self, input, length):
        embeded = tnn.utils.rnn.pack_padded_sequence(
            input,
            length.cpu(),
            batch_first=True
        )
        output, (hidden, cell) = self.lstm(embeded)
        output_rate = self.linear_f_rating2(
            self.relu(self.linear_f_rating1(hidden[1])))
        output_category = self.linear_f_category_2(
            self.relu(self.linear_f_category_1(hidden[1])))
        return output_rate, output_category


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
