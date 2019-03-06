import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Loading the iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names


# loop over k
meanScores = []
for k in range(5,20,2):
    # defining the nearest neighbor classifier
    kNN = KNeighborsClassifier(k, weights='uniform')

    # 5-fold cross validation
    scores = cross_val_score(kNN, X, y, cv=5)
    meanScores.append(scores.mean())


# plotting the mean score vs k
plt.plot(np.arange(5,20,2), meanScores, 'o-')
plt.xlabel('k')
plt.ylabel('Mean score')
plt.show()



# using a grid search
param = {'n_neighbors':list(range(5,20,2)),
         'weights':['uniform', 'distance']}
kNN = KNeighborsClassifier()
grid_kNN = GridSearchCV(kNN, param, cv=5)
grid_kNN.fit(X,y)
print(grid_kNN.best_params_)
print(grid_kNN.best_score_)




# Checking the winning combination
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=2018)
kNN = KNeighborsClassifier(7, weights='uniform')
kNN.fit(X_train,y_train)
y_pred = kNN.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred,
                            target_names=target_names))

