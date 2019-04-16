import nltk

# loading the Gutenberg corpus
from nltk.corpus import gutenberg

# loading Emma
emmaWords = gutenberg.words('austen-emma.txt')

# Lower case
emmaWordsLower = [w.lower() for w in emmaWords]

# Word frequency
emmaDist = nltk.FreqDist(emmaWordsLower)

# Set of unique words
emmaWordSet = sorted(set(emmaWordsLower))

# long words appearing more than 20 times
longFreqWords = [w for w in emmaWordSet
                 if (len(w)>9) and emmaDist[w]>20]
print(longFreqWords)
