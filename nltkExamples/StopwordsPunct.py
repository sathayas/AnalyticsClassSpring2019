import nltk
from nltk.corpus import stopwords

# loading the Emma by Jane Austen
from nltk.corpus import gutenberg
rawText = gutenberg.raw('austen-emma.txt')

# tokenizing
sentText = nltk.sent_tokenize(rawText)
print(sentText[5])
wordText = nltk.word_tokenize(sentText[5])

# removing punctuation marks, making all words lower case
wordDePunct = [w.lower() for w in wordText if w.isalpha()]
print(wordDePunct)

# removing stopwords
stop_words = set(stopwords.words('english'))  # stop words in English
wordNoStopwd = [w for w in wordDePunct if w not in stop_words]
print(wordNoStopwd)


