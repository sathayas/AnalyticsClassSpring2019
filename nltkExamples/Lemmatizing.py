import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# sample words
sampleWords = ['cats','cacti','geese','rocks','oxen','ran','spamming',
               'spammed','spammer','moves','movement','better']
for w in sampleWords:
    print(w)


# lemmatizer object
lmt = WordNetLemmatizer()

# lemmatized words
for w in sampleWords:
    print(lmt.lemmatize(w))

# some non-noun words
print(lmt.lemmatize('ran', pos='v'))
print(lmt.lemmatize('better', pos='a'))



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

# before lemmatizing
print(wordNoStopwd)

# after lemmatizing
wordLemma = [lmt.lemmatize(w) for w in wordNoStopwd]
print(wordLemma)

# using POS tags
wordPOS = nltk.pos_tag(wordText)
# removing punctuation marks & stop words, making all words lower case, 
wordPOSDePunct = [(w[0].lower(), w[1]) for w in wordPOS if w[0].isalpha()]
wordPOSNoStopwd = [w for w in wordPOSDePunct if w[0] not in stop_words]
# initializing the lammatized word list
wordPOSLemma = []
for wPair in wordPOSNoStopwd:
    if wPair[1][0] == 'J':   # i.e., adjectives
        wordPOSLemma.append(lmt.lemmatize(wPair[0],pos='a'))
    elif wPair[1][0] == 'V':  # i.e., verbs
        wordPOSLemma.append(lmt.lemmatize(wPair[0],pos='v'))
    elif 'RB' in wPair[1]:  # i.e., adverbs
        wordPOSLemma.append(lmt.lemmatize(wPair[0],pos='r'))
    else:
        wordPOSLemma.append(lmt.lemmatize(wPair[0]))
print(wordPOSLemma)

        
