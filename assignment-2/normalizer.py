import regex as re
from get_files import get_files

def normalize(text):
    sentences = re.split("(?<!Mr)(?<!Mrs)[\.\?!\”]\s+(?=[A-Z])", text)
    normalized = ""
    for sentence in sentences:
        sentence = re.sub("[[:punct:]]", "", sentence).lower()
        normalized += "<s> {} </s>".format(sentence)
    return normalized

files = get_files(".", ".txt")
books = {}
for file in files:
    text = open(file).read()
    sentences = normalize(text)
    books[file] = sentences

print(books)





