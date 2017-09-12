import pickle
from tokenize import tokenize
import sys

total_unigrams = 23179
total_bigrams = 323535

unigram = pickle.load(open("harry-unigram.gr", "rb"))
bigram = pickle.load(open("harry-bigram.gr", "rb"))

n = int(sys.argv[1])

def prob(gram):
    if n == 1:
        return unigram[gram] / total_unigrams
    else:
        return bigram(gram) / unigram(gram[0])

def likelihood(sentence, gram, n):
    words = tokenize(sentence)
    PS = prob(gram)
    del words[0]
    for word in words:
        PS *= count_gram(word, n)
