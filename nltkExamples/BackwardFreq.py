import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Loading The Adventures of Sherlock Holmes by Arthur Conan Doyle
# from the Project Gutenberg
from urllib import request
url = "http://www.gutenberg.org/ebooks/1661.txt.utf-8"
response = request.urlopen(url)
rawText = response.read().decode('utf8')


# tokenizing
wordText = nltk.word_tokenize(rawText)

# word frequency before doing fancy text processing stuff
wordFreqBefore = nltk.FreqDist(wordText)

# list of variants of 'think'
thinkList = ['think', 'Think', 'thinks', 'Thinks', 'thought', 'Thought',
             'Thoughts','thoughts', 'thinking', 'Thinking']

# frequency of these variants
totFreq = 0
for iThink in thinkList:
    print(iThink + ': %d' % wordFreqBefore[iThink])
    totFreq += wordFreqBefore[iThink]
    
# total frequency
print('Total: %d' % totFreq)



### Fancy text processing
stop_words = set(stopwords.words('english'))  # stop words in English
# Lemmatizing using POS tags
lmt = WordNetLemmatizer()
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

# word frequency after processing        
wordFreqAfter = nltk.FreqDist(wordPOSLemma)
print('After lemmatizing')

# frequency of these variants, lemmatized
totFreq = 0
for iThink in thinkList:
    print(iThink + ': %d' % wordFreqAfter[iThink])
    totFreq += wordFreqAfter[iThink]
    
# total frequency
print('Total: %d' % totFreq)
