import sys
import regex as re
import pickle
import os

"""
Assignment 1 in EDAN20 Language Technology at LTH.
A document comparator using the TF-IDF representation of each document to determine their
similarity. 

Usage: python indexer.py Selma
"""


def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files

def find_words(file):
    """
    Finds the indices of each word in a given file.
    :param file: A text file
    :return: A dict with the word as key and its indices in a list as value. 
    """
    words = re.finditer(r"\w+", open("Selma/" + file, "r").read())
    indices = {}
    for word in words:
        w = word.group().lower()
        if w in indices:
            indices[w].append(word.start())
        else:
            indices[w] = [word.start()]
    return indices

def create_master(indices):
    words = {word for d in indices for word in d[0]}
    master = {}
    for word in words:
        word_indices = {}
        for d in indices:
            try:
                word_indices[d[1]] = d[0][word]
            except KeyError:
                continue
        master[word] = word_indices
    return master

def indexer():
    folder = sys.argv[1]
    files = get_files(folder, "txt")
    indices = []
    for file in files:
        words = find_words(file)
        write_to_file(words, file.replace("txt", "idx"))
        indices.append((words, file))
    master = create_master(indices)
    write_to_file(master, "master.idx")
    print("{} words indexed!".format(len(master)))

def write_to_file(d, file):
    """
    Writes a dict to file using pickle. 
    :param d: The dict to write from. 
    """
    pickle.dump(d, open(file, "wb"))

indexer()