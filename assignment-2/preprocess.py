from get_files import get_files
import regex as re


books = get_files(".", "txt")
for book in books:
    f = open(book, "r")
    text = f.read()
    f = open(book, "w")
    text = re.sub("Page \| \d+ Harry Potter and the " + book.replace(".txt", "") + " - J\.K\. Rowling", "", text)
    text = re.sub("\n+", "", text)
    f.write(text)
