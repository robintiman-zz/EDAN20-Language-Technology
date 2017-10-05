"""
CoNLL-X and CoNLL-U file readers and writers
"""
__author__ = "Pierre Nugues"

import os
import collections

def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    Recursive version
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        path = dir + '/' + file
        if os.path.isdir(path):
            files += get_files(path, suffix)
        elif os.path.isfile(path) and file.endswith(suffix):
            files.append(path)
    return files


def read_sentences(file):
    """
    Creates a list of sentences from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


def split_rows(sentences, column_names):
    """
    Creates a list of sentence where each sentence is a list of lines
    Each line is a dictionary of columns
    :param sentences:
    :param column_names:
    :return:
    """
    new_sentences = []
    root_values = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '0', 'ROOT', '0', 'ROOT']
    start = [dict(zip(column_names, root_values))]
    for sentence in sentences:
        rows = sentence.split('\n')
        sentence = [dict(zip(column_names, row.split())) for row in rows if row[0] != '#']
        sentence = start + sentence
        new_sentences.append(sentence)
    return new_sentences


def save(file, formatted_corpus, column_names):
    f_out = open(file, 'w')
    for sentence in formatted_corpus:
        for row in sentence[1:]:
            # print(row, flush=True)
            for col in column_names[:-1]:
                if col in row:
                    f_out.write(row[col] + '\t')
                else:
                    f_out.write('_\t')
            col = column_names[-1]
            if col in row:
                f_out.write(row[col] + '\n')
            else:
                f_out.write('_\n')
        f_out.write('\n')
    f_out.close()

def pairs(new_sentences):
    pairs = {}
    for sentence in new_sentences:
        for line in sentence:
            if line['deprel'] == 'SS':
                verb_id = int(line['head'])
                verb_line = sentence[verb_id]
                key = (line['form'].lower(), verb_line['form'].lower())
                if key in pairs:
                    pairs[key] += 1
                else:
                    pairs[key] = 1


    return pairs

def triplets(sentences):
    triplets = {}
    for sentence in sentences:
        subjects = {}
        objects = {}
        verb_id_objects = []
        verb_id_subjects = []
        for line in sentence:
            if line['deprel'] == 'OO':
                verb_id_objects.append((line['head'], line['form']))

            if line['deprel'] == 'SS':
                verb_id_subjects.append((line['head'], line['form']))

        for verb_id_object in verb_id_objects:
            for verb_id_subject in verb_id_subjects:
                if verb_id_object[0] == verb_id_subject[0]:
                    verb_line = sentence[int(verb_id_subject[0])]
                    key = (verb_id_subject[1].lower(), verb_line['form'].lower(), verb_id_object[1].lower())
                    if key in triplets:
                        triplets[key] += 1
                    else:
                        triplets[key] = 1

    return triplets


if __name__ == '__main__':
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']

    test_file = 'talbanken.txt'
    sentences = read_sentences(test_file)
    new_sentences = split_rows(sentences, column_names_2006)
    stats = os.stat('talbanken.txt')
    pairs = pairs(new_sentences)
    triplets = triplets(new_sentences)
    pairs = collections.Counter(pairs)
    triplets = collections.Counter(triplets)
    print("Totalt antal SS in corpus: {0}".format(sum(pairs.values())))
    print("Most common pairs: {0}".format(pairs.most_common(5)))
    print("Totalt antal OO in corpus: {0}".format(sum(triplets.values())))
    print("Most common triplets: {0}".format(triplets.most_common(5)))




