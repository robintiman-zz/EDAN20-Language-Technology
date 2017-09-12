import pickle
import sys
import regex as re

def tokenize(text):
    words = re.findall('\p{L}+', text)
    return words


def count_unigrams(words):
    frequency = {}
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1
    return frequency

def count_bigrams(words):
    bigrams = [tuple(words[inx:inx + 2])
               for inx in range(len(words) - 1)]
    frequencies = {}
    for bigram in bigrams:
        if bigram in frequencies:
            frequencies[bigram] += 1
        else:
            frequencies[bigram] = 1
    return frequencies

text = sys.stdin.read().lower()
words = tokenize(text)
uni = count_unigrams(words)
bi = count_bigrams(words)
pickle.dump(uni, open("harry-unigram.gr", "wb"))
pickle.dump(bi, open("harry-bigram.gr", "wb"))
