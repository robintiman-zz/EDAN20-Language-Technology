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


def vectorizer():
    """
    For each key (N) in master. Calculate the tf-idf for that word in each document.
    Create a Nx1 vector for each document and store the calculated values there in.  
    :return: 
    """
    master = load_dicts("master.idx")
    files = get_files(".", "idx")
    files.remove("master.idx") # Don't need it
    dicts = {file.replace("idx", "txt") : load_dicts(file) for file in files}
    vectors = {file.replace("idx", "txt") : {} for file in files}
    total_documents = len(dicts)
    total_terms = {}
    for doc in dicts:
        words = dicts[doc]
        total = sum([len(d) for d in list(words.values())])
        total_terms[doc] = total

    for word in master:
        doc_dict = master[word]
        documents_with_word = len(doc_dict)
        for doc in dicts:
            try:
                times_word = len(doc_dict[doc])
            except KeyError:
                times_word = 0
            if doc == "jerusalem.txt" and word == "nils":
                print("")
            vectors[doc][word] = tfidf(times_word, total_terms[doc], total_documents,
                                       documents_with_word)
    print("Vectorized!")
    results = [vectors[doc]["nils"] for doc in ["bannlyst.txt", "herrgard.txt", "jerusalem.txt", "nils.txt"]]
    print(results)

vectorizer()