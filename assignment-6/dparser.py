"""
Gold standard parser
"""
__author__ = "Pierre Nugues"

from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
import pickle
import features
import transition
import conll

def reference(stack, queue, graph):
    """
    Gold standard parsing
    Produces a sequence of transitions from a manually-annotated corpus:
    sh, re, ra.deprel, la.deprel
    :param stack: The stack
    :param queue: The input list
    :param graph: The set of relations already parsed
    :return: the transition and the grammatical function (deprel) in the
    form of transition.deprel
    """
    # Right arc
    if stack and stack[0]['id'] == queue[0]['head']:
        # print('ra', queue[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + queue[0]['deprel']
        stack, queue, graph = transition.right_arc(stack, queue, graph)
        return stack, queue, graph, 'ra' + deprel
    # Left arc
    if stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        stack, queue, graph = transition.left_arc(stack, queue, graph)
        return stack, queue, graph, 'la' + deprel
    # Reduce
    if stack and transition.can_reduce(stack, graph):
        for word in stack:
            if (word['id'] == queue[0]['head'] or
                        word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                stack, queue, graph = transition.reduce(stack, queue, graph)
                return stack, queue, graph, 're'
    # Shift
    # print('sh', [], queue[0]['cpostag'])
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'

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


def parse(file, save_as,num_features=None, classifier=None, le=None, vec=None):
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    feature_names = [("stack", 0, "postag", "stack0_POS"), ("queue", 0, "postag", "queue0_POS"),
                     ("stack", 0, "form", "stack0_word"), ("queue", 0, "form", "queue0_word"),
                     ("stack", 1, "postag", "stack1_POS"), ("stack", 1, "form", "stack1_word"),
                     ("queue", 1, "postag", "queue1_POS"), ("queue", 1, "form", "queue1_word"),
                     # Here the index is in id or head
                     ("sentence", ("stack", 0, "id"), "postag", "stack0_fw_POS"),
                     ("sentence", ("stack", 0, "id"), "form", "stack0_fw_word"),
                     ("sentence", ("stack", 1, "head"), "postag", "stack0_head_POS"),
                     ("sentence", ("stack", 1, "head"), "form", "stack0_head_word")]

    feature_names = feature_names[:num_features]

    sentences = conll.read_sentences(file)
    if classifier is None or vec is None or le is None:
        formatted_corpus = conll.split_rows(sentences, column_names_2006)
    else:
        formatted_corpus = conll.split_rows(sentences, column_names_2006_test)

    sent_cnt = 0
    X = []
    y = []

    predicted_corpus = []
    for sentence in formatted_corpus:
        sent_cnt += 1
        if sent_cnt % 1000 == 0:
            print(sent_cnt, 'sentences on', len(formatted_corpus), flush=True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'

        while queue:

            if classifier is None or vec is None or le is None:
                feat = features.extract(stack, queue, graph, feature_names, sentence)
                stack, queue, graph, trans = reference(stack, queue, graph)
            else:
                feat = features.extract(stack, queue, graph, feature_names, sentence)
                encode_feat = vec.transform(feat)
                trans_nr = classifier.predict(encode_feat)
                trans = le.inverse_transform(trans_nr)
                stack, queue, graph, trans = parse_ml(stack, queue, graph, trans)

            X.append(feat)
            y.append(trans)

        stack, graph = transition.empty_stack(stack, graph)
        # print('Equal graphs:', transition.equal_graphs(sentence, graph))

        # Poorman's projectivization to have well-formed graphs.
        for word in sentence:

            try:
                word['head'] = graph['heads'][word['id']]
                word['phead'] = graph['heads'][word['id']]
            except KeyError:
                word['head'] = '_'
                word['phead'] = '_'

            try:
                word['deprel'] = graph['deprels'][word['id']]
                word['pdeprel'] = graph['pdeprel'][word['id']]
            except KeyError:
                word['deprel'] = '_'
                word['pdeprel'] = '_'

        predicted_corpus.append(sentence)

    conll.save("{0}.txt".format(save_as), predicted_corpus, column_names_2006)

    return X, y


if __name__ == '__main__':

    feature_lengths = [4]
    for num_features in feature_lengths:
        train_file = 'swedish_talbanken05_train.conll'
        test_file = 'swedish_talbanken05_test_blind.conll'
        X_train, y_train = parse(train_file, "train_" + str(num_features), num_features)
        le = preprocessing.LabelEncoder()
        le.fit_transform(y_train)
        vec = DictVectorizer(sparse=True)
        vec.fit_transform(X_train)

        print("Loading model...")
        with open("../assignment-5/model_" + str(num_features), "rb") as f:
            classifier = pickle.load(f)
        save = "output_{0}".format(str(num_features))
        X_test, y_test = parse(test_file, save, num_features, classifier, le, vec)

