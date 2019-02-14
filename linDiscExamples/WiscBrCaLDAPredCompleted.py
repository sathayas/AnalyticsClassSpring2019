import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# loading the data, extracting features and targets
BrCaData = pd.read_csv('WiscBrCa_clean.csv')
BrCaFeatures = np.array(BrCaData.iloc[:,1:10])
BrCaTargets = np.int_(np.array(BrCaData.iloc[:,10]) /2 -1)  # 0: benign, 1: malignant
featureNames = np.array(BrCaData.columns[1:10])
targetNames = ['Benign','Malignant']


# spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(BrCaFeatures,
                                                    BrCaTargets,
                                                    test_size=0.33,
                                                    random_state=0)

# Fitting the LDA to the training data
BrCaLDA = LinearDiscriminantAnalysis(n_components=1)
X_train_LDA = BrCaLDA.fit_transform(X_train,y_train)

# contribution of different features
print(featureNames)
print(BrCaLDA.scalings_)


# Classification on the testing data
X_test_LDA = BrCaLDA.transform(X_test)
y_pred = BrCaLDA.predict(X_test)

# Confusion matrix
print(confusion_matrix(y_test,y_pred))

# classification report
print(classification_report(y_test, y_pred, target_names=targetNames))
