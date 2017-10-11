from sklearn.linear_model import LogisticRegression
from dparser import reference
import transition
import conll

def extract(stack, queue, graph, feature_names, sentence):
    features = []
    for feature in feature_names:
        stack_or_queue = eval(feature[0])
        index = feature[1]
        tag = feature[2]
        try:
            features.append(stack_or_queue[index][tag])
        except IndexError:
            features.append("nil")

    features.append(transition.can_reduce(stack, graph))
    features.append(transition.can_leftarc(stack, graph))
    stack, queue, graph, action = reference(stack, queue, graph)
    return features, action, stack, queue, graph

train_file = 'swedish_talbanken05_train.conll'
test_file = 'swedish_talbanken05_test_blind.conll'
column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']
feature_names = [("stack", 0, "postag"), ("stack", 1 ,"postag"), ("stack", 0, "form"), ("stack", 1, "form"),
                 ("queue", 0, "postag"), ("queue", 1, "postag"), ("queue", 0, "form"), ("queue", 1, "form")]
                 # "can_re", "can_la"]

sentences = conll.read_sentences(train_file)
formatted_corpus = conll.split_rows(sentences, column_names_2006)

sent_cnt = 0
X = []
y = []
for sentence in formatted_corpus:
    sent_cnt += 1
    if sent_cnt % 1000 == 0:
        print(sent_cnt, 'sentences of', len(formatted_corpus), flush=True)
        break
    stack = []
    queue = list(sentence)
    graph = {}
    graph['heads'] = {}
    graph['heads']['0'] = '0'
    graph['deprels'] = {}
    graph['deprels']['0'] = 'ROOT'
    while queue:
        features, action, stack, queue, graph = extract(stack, queue, graph, feature_names, sentence)
        y.append(action)
        X.append(features)
    stack, graph = transition.empty_stack(stack, graph)
