import regex as re

def split_into_sentences(text):
    # Regex (?<!Mrs)(?<!Mr)(\.|\?|!)
    return re.split(r"(?<!Mr)(?<!Mrs)[\.\?!\”]\s+(?=[A-Z])", text)

def preprocess(text):
    pass

file = open("Book 1 - The Philosopher's Stone_djvu.txt", encoding="utf8")
text = file.read()
print(split_into_sentences(text))

