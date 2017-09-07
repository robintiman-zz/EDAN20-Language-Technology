import pickle
import math
from indexer import get_files
from scipy.spatial.distance import cosine
import numpy as np
from texttable import Texttable

"""
TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
Value = TF * IDF
"""

def load_dicts(file):
    doc = pickle.load(open(file, "rb"))
    return doc


def tfidf(times_word, total_terms, total_documents, documents_with_word):
    tf = times_word / total_terms
    idf = math.log10(total_documents / documents_with_word)
    return tf * idf


def find_total_terms(dicts):
    total_terms = {}
    for doc in dicts:
        words = dicts[doc]
        total = sum([len(d) for d in list(words.values())])
        total_terms[doc] = total
    return total_terms


def vectorizer():
    """
    For each key (N) in master. Calculate the tf-idf for that word in each document.
    Create a Nx1 vector for each document and store the calculated values there in.  
    """
    master = load_dicts("master.idx")
    files = get_files(".", "idx")
    files.remove("master.idx") # Don't need it
    dicts = {file.replace("idx", "txt") : load_dicts(file) for file in files}
    vectors = {file.replace("idx", "txt") : {} for file in files}
    total_documents = len(dicts)
    total_terms = find_total_terms(dicts)
    for word in master:
        doc_dict = master[word]
        documents_with_word = len(doc_dict)
        for doc in dicts:
            try:
                times_word = len(doc_dict[doc])
            except KeyError:
                times_word = 0
            vectors[doc][word] = tfidf(times_word, total_terms[doc], total_documents,
                                       documents_with_word)

    print("{} words vectorized!".format(len(master)))
    return vectors


def cosine_similarity(vectors):
    """
    cosine similarity = 1 - cosine distance 
    :return: Matrix S(i,j) = The similarity between document i and j
    """
    length = len((vectors))
    S = np.zeros((length, length))
    docs = list(vectors.keys())
    for i in range(length):
        for j in range(i + 1, length):
            doc_i = list(vectors[docs[i]].values())
            doc_j = list(vectors[docs[j]].values())
            S[i, j] = S[j, i] =  1 - cosine(doc_i, doc_j)
    index = list(np.argsort(S, axis=None))
    np.set_printoptions(5, suppress=True)
    print_matrix(S, docs)
    print("The most similar are:")
    j = 1
    for i in range(len(index)-1, length, -2):
        doc_index = np.unravel_index(index[i], (length, length))
        print("{}. {} and {}".format(j, docs[doc_index[0]], docs[doc_index[1]]))
        j += 1

def print_matrix(S, docs):
    table = Texttable()
    header = [""] + docs
    table.add_row(header)
    for i in range(len(S)):
        row = [docs[i]] + list(S[i, :])
        table.add_row(row)
    print(table.draw())

cosine_similarity(vectorizer())