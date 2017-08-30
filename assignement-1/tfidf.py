import pickle

"""
TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
Value = TF * IDF
"""

def load_files():
    master = pickle.load(open("master.idx", "rb"))
    return master

def tfidf():
    master = load_files()
    

tfidf()