import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# loading the digits data
digits = datasets.load_digits()
digitsX = digits.data    # the data, 1797 x 64 array
digitsTargets = digits.target # target information
digitsTargetNames = [str(digits.target_names[i]) 
                     for i in range(len(digits.target_names))]  # digits

# dimension reduction with PCA, with 13 PCs
digitsPCA = PCA(n_components=13)
digitsPCs = digitsPCA.fit_transform(digitsX)


# spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(digitsPCs,
                                                    digitsTargets, 
                                                    test_size=0.3,
                                                    random_state=0)

# Fitting the logisic regression 
digitsLR = LogisticRegression()
digitsLR.fit(X_train,y_train)

# Classification on the testing data
y_pred_LR = digitsLR.predict(X_test)

# Confusion matrix
print(confusion_matrix(y_test,y_pred_LR))

# classification report
print(classification_report(y_test, y_pred_LR, target_names=digitsTargetNames))

