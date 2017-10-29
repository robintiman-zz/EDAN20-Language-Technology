"""
Gold standard parser
"""
__author__ = "Pierre Nugues"


from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn.metrics import classification_report
import sys
import features
import transition
import conll
import pickle


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

def parse(file, save_as, num_features):
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    # can-re and can-la are added manually in extract()
    feature_names = [("stack", 0, "postag", "stack0_POS"), ("queue", 0, "postag", "queue0_POS"),
                     ("stack", 0, "form", "stack0_word"), ("queue", 0, "form", "queue0_word"),
                     ("stack", 1, "postag", "stack1_POS"), ("stack", 1, "form", "stack1_word"),
                     ("queue", 1, "postag", "queue1_POS"), ("queue", 1, "form", "queue1_word"),
                     # Here the index is in id or head
                     ("sentence", (("stack", 0, "id"), 1), "postag", "sent_fw_POS"),
                     ("sentence", (("stack", 0, "id"), 1), "form", "sent_fw_word"),
                     ("sentence", (("queue", 0, "id"), -1), "postag", "stack0_head_POS"),
                     ("sentence", (("queue", 0, "id"), -1), "form", "stack0_head_word")]

    feature_names = feature_names[:num_features]

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    sent_cnt = 0
    X = []
    y = []
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
        while queue:
            feat = features.extract(stack, queue, graph, feature_names, sentence)
            stack, queue, graph, trans = reference(stack, queue, graph)

            X.append(feat)
            y.append(trans)
        stack, graph = transition.empty_stack(stack, graph)
        # Poorman's projectivization to have well-formed graphs.
        for word in sentence:
            word['head'] = graph['heads'][word['id']]
        if not transition.equal_graphs(sentence, graph):
            print(sentence)
            print(graph)

    with open("X_{}.data".format(save_as), "wb") as f:
        pickle.dump(X, f)

    with open("y_{}.data".format(save_as), "wb") as f:
        pickle.dump(y, f)

    return X, y


def save_file(file_name, obj):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)

if __name__ == '__main__':
    do_parsing = True
    do_save = True
    train_file = 'swedish_talbanken05_train.conll'
    test_file = 'swedish_talbanken05_test.conll'
    num_features = int(sys.argv[1])
    if do_parsing:
        X_train, y = parse(train_file, "train_" + str(num_features), num_features)
        X_test, y_test = parse(test_file, "test_" + str(num_features), num_features)
    else:
        with open("X_train_{}.data".format(num_features), "rb") as f:
            X_train = pickle.load(f)
        with open("y_train_{}.data".format(num_features), "rb") as f:
            y = pickle.load(f)

    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_train)

    save_file("vec_{}".format(num_features), vec)

    print("Training model...")
    classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
    model = classifier.fit(X, y)

    print("Predicting..")
    y_pred = classifier.predict(X)
    print(classification_report(y, y_pred))

    if do_save:
        print("Saving model..")
        with open("othermodel_" + str(num_features), "wb") as f:
            pickle.dump(classifier, f)

    print("Done!")
