import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# loading the data, creating binary target
BrCaData = pd.read_csv('WiscBrCa_clean.csv')
BrCaData['Malignant'] = BrCaData.Class //2 -1  # Binary variable of malignancy
                                               # 0: benign
                                               # 1: malignant

# means according to malignancy
print(BrCaData.groupby('Malignant').mean())

# data for logistic regression
BrCaFeatures = np.array(BrCaData.iloc[:,1:10])
BrCaTargets = np.array(BrCaData.Malignant)
featureNames = np.array(BrCaData.columns[1:10])
targetNames = ['Benign','Malignant']


# spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(BrCaFeatures, BrCaTargets,
                                                    test_size=0.33, random_state=0)

# Fitting the logistic regression to the training data
BrCaLR = LogisticRegression()
BrCaLR.fit(X_train,y_train)


# Printing out the odds ratios
print('Feature    \tOdds Ratio')
for i,iFeature in enumerate(featureNames):
    print('%-12s' % iFeature, end='')
    # Odds ratio associated with each feature (unit increase)
    print('\t%8.3f' % np.exp(BrCaLR.coef_[0,i]))



# Classification on the testing data
y_pred = BrCaLR.predict(X_test)
y_prob = BrCaLR.predict_proba(X_test)
plt.plot(y_prob[:,1],y_pred,'b.')
plt.ylim([-0.05, 1.05])
plt.xlabel('Predicted probability')
plt.ylabel('Classification')
plt.show()

# Confusion matrix
print(confusion_matrix(y_test,y_pred))

# classification report
print(classification_report(y_test, y_pred, target_names=targetNames))
