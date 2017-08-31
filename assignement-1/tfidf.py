import pickle
import math
from indexer import get_files

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


def get_constants():
    files = get_files(".", "idx")


def vectorizer():
    """
    For each key (N) in master. Calculate the tf-idf for that word in each document.
    Create a Nx1 vector for each document and store the calculated values there in.  
    :return: 
    """
    master = load_dicts("master.idx")
    files = get_files(".", "idx")
    files.remove("master.idx") # Don't need it
    dicts = {file : load_dicts(file) for file in files}
    vectors = {file : [0] * len(master) for file in files}
    del files
    i = 0
    for word in master:
        doc_dict = master[word]
        for doc in doc_dict:
            times_word = len(doc_dict[doc])
            total_terms = len(dicts[doc])
            total_documents = len(dicts)
            documents_with_word = len(doc_dict)
            vectors[doc][i] = tfidf(times_word, total_terms, total_documents,
                                    documents_with_word)
        i += 1


vectorizer()