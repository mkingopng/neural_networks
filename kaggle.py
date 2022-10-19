import re
import numpy as np
import pandas as pd


def clean_text(text):
    text = text.lower()  # lower case characters only
    text = re.sub('http\S+', ' ', text)  # remove urls
    text = re.sub("[^a-z' ]+", ' ', text)  # only alphabets, spaces and apostrophes
    text = ' ' + text + ' '  # remove all apostrophes which are not used in word contractions
    text = re.sub("[^a-z]'|'[^a-z]", ' ', text)
    return text.split()


df['text'] = df['text'].apply(lambda x: clean_text(x))

unknown_words = []
total_words = 0

def find_unknown_words(words):
    global total_words
    total_words = total_words + len(words)
    for word in words:
        if not (word in embeddings_index_200):
            unknown_words.append(word)
    return words


def analyze_unknown_words(unknown_words):
    unknown_words = np.array(unknown_words)
    (word, count) = np.unique(unknown_words, return_counts=True)
    word_freq = pd.DataFrame({'word': word, 'count': count}).sort_values(
        'count', ascending=False)


