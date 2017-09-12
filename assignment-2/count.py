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

if __name__ == '__main__':
    text = sys.stdin.read().lower()
    words = tokenize(text)
    frequency = count_unigrams(words)
    for word in sorted(frequency.keys(), key=frequency.get, reverse=True):
        print(word, '\t', frequency[word])

if __name__ == '__main__':
    text = sys.stdin.read().lower()
    words = tokenize(text)
    frequency_bigrams = count_bigrams(words)
    for bigram in frequency_bigrams:
        print(frequency_bigrams[bigram], "\t", bigram)