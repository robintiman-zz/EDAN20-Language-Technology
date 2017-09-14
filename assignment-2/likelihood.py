import pickle
import sys
import math
import regex as re


def prob(gram, n):
    if n == 1:
        return (unigram[gram] / total_unigrams, unigram[gram])
    else:
        return (bigram[gram] / unigram[gram[0]], bigram[gram], unigram[gram[0]])


def entropy_rate(sentence_prob, length):
    return -1/length * math.log2(sentence_prob)


def perplexity(entropy):
    return 2**entropy


def print_model(words, p, entropy, perplexity):
    if n == 1:
        model = "Unigram model"
        header = "wi C(wi) #words P(wi)"
    else:
        model = "Bigram model"
        header = "wi wi+1 Ci,i+1 C(i) P(wi+1|wi)"
    print("""
    {}\n
    =====================================================
    {}\n
    =====================================================
    
    """.format())


total_unigrams = 23179
total_bigrams = 323535

unigram = pickle.load(open("harry-unigram.gr", "rb"))
bigram = pickle.load(open("harry-bigram.gr", "rb"))

sentence = sys.argv[1]
sentence = re.sub("[[:punct:]]", "",sentence).lower()
words = sentence.split(" ")
n = int(sys.argv[2])


P = {}
w1 = words[0]
pw1 = prob(words[0], 1)
P[w1] = pw1
PS = pw1[0]
for i in range(1, len(words) - n + 1):
    if n == 1:
        gram = words[i]
    else:
        gram = (words[i], words[i+1])

    p = prob(gram, n)
    PS *= p[0]
    P[gram] = p

er = entropy_rate(PS, len(words))
perp = perplexity(er)
print_model(words, PS, er, perp)
