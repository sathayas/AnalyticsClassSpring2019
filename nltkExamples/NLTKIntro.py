import nltk

# downloading example text corpora "book"
nltk.download()

# loading the Gutenberg corpus
from nltk.corpus import gutenberg
gutenberg.fileids()

# loading the raw text
emmaRawText = gutenberg.raw('austen-emma.txt')
print(emmaRawText[:300])

# loading words
emmaWords = gutenberg.words('austen-emma.txt')
print(emmaWords[:80])
