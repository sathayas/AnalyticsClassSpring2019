import nltk
import random

# a function to return a feature to classify whether a name is
# male or female.
# The feature and the label are returned together
def gender_feature(name):
    featureDict = {'last-letter': name[-1]}
    return featureDict

# reading names from the names corpus
from nltk.corpus import names
femaleNames = names.words('female.txt')
maleNames = names.words('male.txt')

# creating name-label pairs, then shuffling
nameData = []
for iName in femaleNames:
    nameData.append((iName, 'female'))
for iName in maleNames:
    nameData.append((iName, 'male'))
random.shuffle(nameData)

# converting the name data into feature (i.e., just the last letter)
# as well as the label (female / male)
featureData = [(gender_feature(n), gender) for (n, gender) in nameData]

# spliting into training and testing data sets
trainData, testData = featureData[1000:], featureData[:1000]

# training a classifier (Naive Bayes)
clf = nltk.NaiveBayesClassifier.train(trainData)

# classification example
print(clf.classify(gender_feature('Nemo')))
print(clf.classify(gender_feature('Dory')))

# classifier performance on the testing data
print(nltk.classify.accuracy(clf, testData))

# most informative features
clf.show_most_informative_features(10)


# examining classification errors
errorData = []
testDataFull = nameData[:1000]   # Extracting the full testing data
for iData in testDataFull:
    trueCat = iData[1]
    predCat = clf.classify(gender_feature(iData[0]))
    if predCat != trueCat:
        errorData.append((trueCat, predCat, iData[0]))


# printing out the errors
for (y_true, y_pred, name) in sorted(errorData):
    print('Truth: %-6s\t' % y_true, end='')
    print('Pred: %-6s\t' % y_pred, end='')
    print('Name: %-12s' % name)
