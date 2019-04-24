import nltk

# loading the file with Amazon reviews
f = open('SentimentReviews/amazon_cells_labelled.txt','r')
reviewList = []
while True:
    lineData = f.readline().strip()
    if lineData:
        cat = int(lineData[-1])
        text = nltk.word_tokenize(lineData[:-1])
        reviewList.append((text, cat))
    else:
        break

# separating into testing and training data sets
trainList, testList = reviewList[200:], reviewList[:200]


# creating a list of all words in the training data set
allWords = []
for iReviewPair in trainList:
    reviewWords = [w.lower() for w in iReviewPair[0]]
    # Just in case someone writes a review IN ALL CAPS
    allWords += reviewWords


# word frequency, and just consider 2000 most frequent words
allWordFreq = nltk.FreqDist(allWords)
featureWords = [w for (w,c) in allWordFreq.most_common(2000)]


# Document features (whether contains certain words)
def document_features(document): 
    document_words = set(document) 
    features = {}
    for w in featureWords:
        features['contains({})'.format(w)] = (w in document_words)
    return features


# extracting features for training and testing data
trainSet = [(document_features(d), c) for (d,c) in trainList]
testSet = [(document_features(d), c) for (d,c) in testList]


# classifier
clf = nltk.NaiveBayesClassifier.train(trainSet)
print(nltk.classify.accuracy(clf, testSet)) 


# most informative features
clf.show_most_informative_features(15)
