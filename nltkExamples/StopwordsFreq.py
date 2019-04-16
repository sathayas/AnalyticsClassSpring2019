import nltk
from nltk.corpus import stopwords

# Loading The Adventures of Sherlock Holmes by Arthur Conan Doyle
# from the Project Gutenberg
from urllib import request
url = "http://www.gutenberg.org/ebooks/1661.txt.utf-8"
response = request.urlopen(url)
rawText = response.read().decode('utf8')

# tokenizing
wordText = nltk.word_tokenize(rawText)

# word frequency before removing punctuations and stop words
print('Before text processing')
wordFreqBefore = nltk.FreqDist(wordText)
for iWord in wordFreqBefore.most_common(30):
    print('%-15s\t%6d' % iWord)


# removing punctuations and stopwords
wordDePunct = [w.lower() for w in wordText if w.isalpha()]
stop_words = set(stopwords.words('english'))  # stop words in English
wordNoStopwd = [w for w in wordDePunct if w not in stop_words]


# word frequency after removing punctuations and stop words
print('After text processing')
wordFreqAfter = nltk.FreqDist(wordNoStopwd)
for iWord in wordFreqAfter.most_common(30):
    print('%-15s\t%6d' % iWord)
