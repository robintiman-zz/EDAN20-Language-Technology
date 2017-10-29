"""
Gold standard parser
"""
__author__ = "Pierre Nugues"

import pickle
import features
import transition
import conll
import sys

feature_names = [("stack", 0, "postag", "stack0_POS"), ("queue", 0, "postag", "queue0_POS"),
                 ("stack", 0, "form", "stack0_word"), ("queue", 0, "form", "queue0_word"),
                 ("stack", 1, "postag", "stack1_POS"), ("stack", 1, "form", "stack1_word"),
                 ("queue", 1, "postag", "queue1_POS"), ("queue", 1, "form", "queue1_word"),
                 # Here the index is in id or head
                 ("sentence", (("stack", 0, "id"), 1), "postag", "sent_fw_POS"),
                 ("sentence", (("stack", 0, "id"), 1), "form", "sent_fw_word"),
                 ("sentence", (("queue", 0, "id"), -1), "postag", "stack0_head_POS"),
                 ("sentence", (("queue", 0, "id"), -1), "form", "stack0_head_word")]



def parse_ml(stack, queue, graph, trans):
    if stack and trans[:2] == 'ra':
        stack, queue, graph = transition.right_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'ra'
    # Left arc
    if stack and trans[:2] == 'la':
        stack, queue, graph = transition.left_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'la'

    # Reduce
    if stack and trans[:2] == 're':
        stack, queue, graph = transition.reduce(stack, queue, graph)
        return stack, queue, graph, 're'

    #shift
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'


def parse(file, classifier, vec, feature_names, save_as):
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    sentences = conll.read_sentences(file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006_test)

    sent_cnt = 0

    for sentence in formatted_corpus:
        sent_cnt += 1
        if sent_cnt % 1000 == 0:
            print(sent_cnt, 'sentences of', len(formatted_corpus), flush=True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'

        # Predict the next transition and then execute it.
        while queue:
            extracted_feature = features.extract(stack, queue, graph, feature_names, sentence)
            feat = vec.transform(extracted_feature)
            trans = classifier.predict(feat)[0]
            stack, queue, graph, trans = parse_ml(stack, queue, graph, trans)

        stack, graph = transition.empty_stack(stack, graph)

        # Poorman's projectivization to have well-formed graphs.
        for word in sentence:
            try:
                word['head'] = graph['heads'][word['id']]
                word['phead'] = '_'
            except KeyError:
                word['head'] = '_'
                word['phead'] = '_'

            try:
                word['deprel'] = graph['deprels'][word['id']]
                word['pdeprel'] = '_'
            except KeyError:
                word['deprel'] = '_'
                word['pdeprel'] = '_'

    conll.save("{0}.txt".format(save_as), formatted_corpus, column_names_2006)

def open_file(file):
    with open(file, "rb") as f:
        return pickle.load(f)

if __name__ == '__main__':
    print("Predicting..")
    num_features = int(sys.argv[1])
    test_file = 'swedish_talbanken05_test_blind.conll'
    classifier = open_file("../assignment-5/othermodel_{}".format(num_features))
    vec = open_file("../assignment-5/vec_{}".format(num_features))
    save_as = "prediction_{}".format(num_features)
    parse(test_file, classifier, vec, feature_names[:num_features], save_as)
    print("Done! Output file saved as {}".format(save_as))

