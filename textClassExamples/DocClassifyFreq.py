import nltk
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# creating a list of document-label pairs
from nltk.corpus import movie_reviews as mr
reviewList = []
for iCat in mr.categories():  # first, going over categories (pos or neg)
    for iReview in mr.fileids([iCat]):   # reviews in that category
        reviewPair = (mr.raw(iReview), iCat)
        reviewList.append(reviewPair)

# splitting into training and testing data
X = [d for (d, c) in reviewList]
Y = [c for (d, c) in reviewList]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=500,
                                                    random_state=0)

# word occurrence counts
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
# List of words
#count_vect.get_feature_names()
# indices for non-zero elements in the sparse matrix
X_train_counts.nonzero()


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
print(classification_report(Y_test,Y_pred_nb))



# classifier (Linear SVM)
clf_svm = LinearSVC().fit(X_train_tf, Y_train)

# classifying the testing data
Y_pred_svm = clf_svm.predict(X_test_tf)

# accuracy
print('Accuracy - Linear SVM: %6.4f' % accuracy_score(Y_test,Y_pred_svm))
print(confusion_matrix(Y_test,Y_pred_svm))
print(classification_report(Y_test,Y_pred_svm))
