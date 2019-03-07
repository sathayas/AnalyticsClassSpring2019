import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Loading the iris data
iris = datasets.load_iris()
X = iris.data 
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# using a grid search
param = {'C':[10, 1.0, 0.1],
         'kernel':['linear', 'rbf', 'poly']}


svm = SVC()
grid_svm = GridSearchCV(svm, param, cv=5)
grid_svm.fit(X,y)


print(grid_svm.best_params_)
print(grid_svm.best_score_)
