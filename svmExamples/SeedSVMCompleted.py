import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# loading the data 
seedData = pd.read_csv('seeds_dataset.txt', sep='\t', header=None)
seedFeatures = np.array(seedData.iloc[:,:7])
seedTargets = np.array(seedData.iloc[:,7]) - 1 # starting from zero
targetNames = ['Kama','Rosa','Canadian']


# splitting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(seedFeatures,
                                                    seedTargets, 
                                                    test_size=70, 
                                                    random_state=0)

# standardizing the training and testing data sets
normTrain = StandardScaler().fit(X_train)
X_train_norm = normTrain.transform(X_train)
X_test_norm = normTrain.transform(X_test)



# SVM on training data (RBF)
print('Seed data SVM, with RBF kernel')
svRBF = SVC(kernel='rbf',C=1.0)
svRBF.fit(X_train_norm,y_train)

# Predicted classes (RBF)
y_pred_RBF = svRBF.predict(X_test_norm)

# Confusion matrix (RBF)
print(confusion_matrix(y_test,y_pred_RBF))

# classification report (RBF)
print(classification_report(y_test, y_pred_RBF,
                            target_names=targetNames))
print()



# SVM on training data (Poly)
print('Seed data SVM, with Poly kernel')
svPoly = SVC(kernel='poly',C=1.0)
svPoly.fit(X_train_norm,y_train)

# Predicted classes (Poly)
y_pred_Poly = svPoly.predict(X_test_norm)

# Confusion matrix (Poly)
print(confusion_matrix(y_test,y_pred_Poly))

# classification report (Poly)
print(classification_report(y_test, y_pred_Poly,
                            target_names=targetNames))
print()



# SVM on training data (Linear)
print('Seed data SVM, with linear kernel')
svLin = SVC(kernel='linear',C=1.0)
svLin.fit(X_train_norm,y_train)

# Predicted classes (Linear)
y_pred_Lin = svLin.predict(X_test_norm)

# Confusion matrix (Linear)
print(confusion_matrix(y_test,y_pred_Lin))

# classification report (Linear)
print(classification_report(y_test, y_pred_Lin,
                            target_names=targetNames))
print()



