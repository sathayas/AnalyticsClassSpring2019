import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report


# loading the data 
seedData = pd.read_csv('seeds_dataset.txt', sep='\t', header=None)
seedFeatures = np.array(seedData.iloc[:,:7])
seedTargets = np.array(seedData.iloc[:,7]) - 1 # starting from zero
targetNames = ['Kama','Rosa','Canadian']


# pipeline of transformations
svm = Pipeline([
    ('normalize',StandardScaler()),
    ('classify',SVC())
])
param = {'classify__C': [10,1,0.1],
         'classify__kernel': ['rbf','poly','linear']}


# grid search
grid_svm = GridSearchCV(svm, param_grid=param, cv=10)
grid_svm.fit(seedFeatures,seedTargets)
print(grid_svm.best_params_)
print(grid_svm.best_score_)




# Checking the winning combination
X_train, X_test, y_train, y_test = train_test_split(seedFeatures,
                                                    seedTargets, 
                                                    test_size=70, 
                                                    random_state=0)
normTrain = StandardScaler().fit(X_train)
X_train_norm = normTrain.transform(X_train)
X_test_norm = normTrain.transform(X_test)
sv = SVC(kernel='linear',C=10)
sv.fit(X_train_norm,y_train)
y_pred = sv.predict(X_test_norm)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred,
                            target_names=targetNames))

