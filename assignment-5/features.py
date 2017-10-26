import transition


def extract(stack, queue, graph, feature_names, sentence):
    features = {}
    for feature in feature_names:
        index, name, data, tag = get_params(feature, stack, queue, sentence)
        try:
            features[name] = data[index][tag]
        except IndexError:
            features[name] = "nil"
    features["can-re"] = transition.can_reduce(stack, graph)
    features["can-la"] = transition.can_leftarc(stack, graph)
    features["can-ra"] = transition.can_rightarc(stack)
    return features

# ("sentence", ("stack", 0, "id"), "postag", "stack0_fw_POS"),


def get_params(feature, stack, queue, sentence):
    index = feature[1]
    if not isinstance(index, int):
        p = index[0]
        fw = index[1]
        data_t = eval(p[0])
        index_t = p[1]
        tag_t = p[2]
        try:
            index = int(data_t[index_t][tag_t]) + fw
        except IndexError:
            index = -1
    data = eval(feature[0])
    tag = feature[2]
    name = feature[3]
    return index, name, data, tag
