import nltk

# Gutenberg corpus
from nltk.corpus import gutenberg
print(gutenberg.fileids())

# Movie reviews
from nltk.corpus import movie_reviews as mr
#print(mr.fileids())
print(mr.raw('neg/cv981_16679.txt')[:300])
print(mr.raw('pos/cv997_5046.txt')[:300])

# Brown corpus
from nltk.corpus import brown
print(brown.categories())
print(brown.fileids(['news'])[:20])
print(brown.raw(fileids='ca34')[:300])

# Reuter corpus
from nltk.corpus import reuters
print(reuters.categories())
print(reuters.fileids(['housing']))
print(reuters.raw(fileids='training/3720')[:300])

