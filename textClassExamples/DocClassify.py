import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC

# creating a list of document-label pairs
from nltk.corpus import movie_reviews as mr
reviewList = []
for iCat in mr.categories():  # first, going over categories (pos or neg)
    for iReview in mr.fileids([iCat]):   # reviews in that category
        reviewPair = (mr.words(iReview), iCat)
        reviewList.append(reviewPair)

# shuffling, and separating into testing and training data sets
random.shuffle(reviewList)
trainList, testList = reviewList[500:], reviewList[:500]

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



# Now with SVM classifier (linear kernel)
clf_svm = SklearnClassifier(LinearSVC())
clf_svm.train(trainSet)
print(nltk.classify.accuracy(clf_svm, testSet)) 

