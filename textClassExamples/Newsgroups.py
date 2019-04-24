import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# loading the newsgroup data, as training and testing data
newsGroups = ['soc.religion.christian',
              'comp.graphics',
              'sci.med', 
              'rec.sport.baseball']

from sklearn.datasets import fetch_20newsgroups
newsTrain = fetch_20newsgroups(subset='train',
                               categories=newsGroups, 
                               shuffle=True, 
                               random_state=0)
newsTest = fetch_20newsgroups(subset='test',
                              categories=newsGroups, 
                              shuffle=True, 
                              random_state=0)

X_train = newsTrain.data
Y_train = newsTrain.target
X_test = newsTest.data
Y_test = newsTest.target


# Example news posts
print(X_train[111])
print(X_train[1649])



# Exercise code here!

