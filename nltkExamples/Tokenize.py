import nltk

# Loading The Adventures of Sherlock Holmes by Arthur Conan Doyle
# from the Project Gutenberg
from urllib import request
url = "http://www.gutenberg.org/ebooks/1661.txt.utf-8"
response = request.urlopen(url)
rawText = response.read().decode('utf8')

# breaking up the raw text into sentences
sentText = nltk.sent_tokenize(rawText)

print(sentText[14])
print(sentText[15])
print(sentText[16])

# breaking up sentences into words
print(nltk.word_tokenize(sentText[14]))
print(nltk.word_tokenize(sentText[15]))
print(nltk.word_tokenize(sentText[16]))


# breaking up the raw text into words
wordText = nltk.word_tokenize(rawText)


# word frequency
wordDist = nltk.FreqDist(wordText)
print(wordDist.most_common(30))

# word frequency after converting to lower case
wordTextLower = [w.lower() for w in wordText]
wordDistLower = nltk.FreqDist(wordTextLower)
print(wordDistLower.most_common(30))


# just long words (10 characters or more)
wordSetLower = sorted(set(wordTextLower))  # unique word list
longWords = [w for w in wordSetLower if len(w)>9]

# long words appearing more than 20 times
longFreqWords = [w for w in wordSetLower
                 if (len(w)>9) and wordDistLower[w]>20]

