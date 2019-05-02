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
#print(X_train[111])
#print(X_train[1649])

# word occurrence counts
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

# converting to term frequency
tf_transformer = TfidfTransformer()
X_train_tf = tf_transformer.fit_transform(X_train_counts)

# classifier (naive Bayes)
clf_nb = MultinomialNB().fit(X_train_tf, Y_train)

# converting the testing set to term frequency
X_test_counts = count_vect.transform(X_test)  # NB you don't have to fit
X_test_tf = tf_transformer.transform(X_test_counts)  # NB you don't have to fit

# classifying the testing data
Y_pred_nb = clf_nb.predict(X_test_tf)

# accuracy
print('Accuracy - Naive Bayes: %6.4f' % accuracy_score(Y_test,Y_pred_nb))
print(confusion_matrix(Y_test,Y_pred_nb))
print(classification_report(Y_test,Y_pred_nb,
                            target_names=newsTest.target_names))



# classifier (Linear SVM)
clf_svm = LinearSVC().fit(X_train_tf, Y_train)

# classifying the testing data
Y_pred_svm = clf_svm.predict(X_test_tf)

# accuracy
print('Accuracy - Linear SVM: %6.4f' % accuracy_score(Y_test,Y_pred_svm))
print(confusion_matrix(Y_test,Y_pred_svm))
print(classification_report(Y_test,Y_pred_svm,
                            target_names=newsTest.target_names))
