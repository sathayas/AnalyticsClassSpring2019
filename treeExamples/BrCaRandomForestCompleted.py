import numpy as np
import pandas as pd
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# loading the data, creating binary target
BrCaData = pd.read_csv('WiscBrCa_clean.csv')
BrCaData['Malignant'] = BrCaData.Class //2 -1  # Binary variable of malignancy
                                               # 0: benign
                                               # 1: malignant


# means according to malignancy
print(BrCaData.groupby('Malignant').mean())


# features and targets
BrCaFeatures = np.array(BrCaData.iloc[:,1:10])
BrCaTargets = np.array(BrCaData.Malignant)
featureNames = np.array(BrCaData.columns[1:10])
targetNames = ['Benign','Malignant']


# spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(BrCaFeatures,
                                                    BrCaTargets,
                                                    test_size=200,
                                                    random_state=0)


# random forest classifier, training & testing
rf = RandomForestClassifier(criterion='entropy',
                            n_estimators = 50,
                            min_samples_leaf = 7,
                            max_depth = 7,
                            random_state=0)
rf.fit(X_train,y_train)


# evaluating the classifier performance
y_pred = rf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred,
                            target_names=targetNames))



# For a comparision, 
# decision tree classifier, training & testing
dt = DecisionTreeClassifier(criterion='entropy', 
                            min_samples_leaf = 7,
                            max_depth = 7,
                            random_state=0)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred,
                            target_names=targetNames))
