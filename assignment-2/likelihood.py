import pickle
import sys
import math
import regex as re


def prob(gram, n):
    if n == 1:
        try:
            count = unigram[gram]
        except KeyError:
            count = 1000
        return count / total_unigrams
    else:
        try:
            bi_count = bigram[gram]
            uni_count = unigram[gram[0]]
            p = bi_count / uni_count
        except KeyError:
            p = prob(gram[1], 1)
        return p


def entropy_rate(sentence_prob, length):
    return -1/length * math.log2(sentence_prob)


def perplexity(entropy):
    return 2**entropy


def print_model(words, P, entropy, perplexity, PS):
    if n == 1:
        model = "Unigram model"
        header = "wi C(wi) #words P(wi)"
    else:
        model = "Bigram model"
        header = "wi wi+1 Ci,i+1 C(i) P(wi+1|wi)"
    print("{}\n=====================================================\n{}\n"
          "=====================================================".format(model, header))
    for i in range(len(words) - n + 1):
        if n == 1:
            word = words[i]
            nbr_words = total_unigrams
            c = unigram[word]
            p = P[word]
            error = ""
        else:
            word = (words[i], words[i+1])
            try:
                nbr_words = "{}".format(unigram[word[0]])
                c = bigram[word]
                p = P[word]
                error = ""
            except KeyError:
                c = "0"
                p = 0.0
                error = "*backoff: {}".format(P[word])

        print("{word} {c} {nbr_words} {p} {error}".format(word=word, c=c,
                                                  nbr_words=nbr_words, p=p, error=error))
    print("Prob grams: {}\nEntropy rate: {}\nPerplexity: {}".format(PS, entropy, perplexity))

total_unigrams = 3552556
total_bigrams = 3552555

unigram = pickle.load(open("harry-unigram.gr", "rb"))
bigram = pickle.load(open("harry-bigram.gr", "rb"))

sentence = sys.argv[1]
sentence = re.sub("[[:punct:]]", "",sentence).lower()
words = sentence.split(" ")
n = int(sys.argv[2])

P = {}
w1 = words[0]
pw1 = prob(words[0], 1)
PS = pw1
for i in range(0, len(words) - n + 1):
    if n == 1:
        gram = words[i]
    else:
        gram = (words[i], words[i+1])

    p = prob(gram, n)
    PS *= p
    P[gram] = p

er = entropy_rate(PS, len(words))
perp = perplexity(er)
print_model(words, P, er, perp, PS)
