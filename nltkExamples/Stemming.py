import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# sample words
sampleWords = ['spam', 'spams', 'spamming', 'spammed', 'spammer', 'spammers',
               'spamize','spamly']

# stemmer object
ps = PorterStemmer()

for w in sampleWords:
    print(ps.stem(w))


# loading the Emma by Jane Austen
from nltk.corpus import gutenberg
rawText = gutenberg.raw('austen-emma.txt')

# tokenizing
sentText = nltk.sent_tokenize(rawText)
wordText = nltk.word_tokenize(sentText[5])

# removing punctuation marks & stop words, making all words lower case, 
wordDePunct = [w.lower() for w in wordText if w.isalpha()]
stop_words = set(stopwords.words('english'))  # stop words in English
wordNoStopwd = [w for w in wordDePunct if w not in stop_words]

# before stemming
print(wordNoStopwd)

# after stemming
wordStem = [ps.stem(w) for w in wordNoStopwd]
print(wordStem)
