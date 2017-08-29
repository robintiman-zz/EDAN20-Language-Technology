import sys
import re
import pickle
import os

"""
Assignment 1 in EDAN20 Language Technology at LTH.
A document comparator using the TF-IDF representation of each document to determine their
similarity. 

Usage: <fill in>
"""

def main():
    folder = sys.argv[0]

    # Using a dictionary with the word as key and all the corresponding indices as value.
    indices = {}


def find_words(text):
    words = re.finditer(r"\w", text)
    print(words.groups())

def write_to_file(indices, file):
    """
    Writes a dict to file using pickle. 
    :param indices: The dict to write from. 
    """
    pickle.dump(indices, open(file, "wb"))


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