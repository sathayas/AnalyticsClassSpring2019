import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# Loading the iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target


# defining the nearest neighbor classifier
kNN = KNeighborsClassifier(5, weights='uniform')

# 5-fold cross validation
scores = cross_val_score(kNN, X, y, cv=5)
print(scores)
print(scores.mean())

