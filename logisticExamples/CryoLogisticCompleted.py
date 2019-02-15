import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# loading the data
CryoData = pd.read_csv('Cryotherapy.csv')

# examining the outcome vs other variables
print(CryoData.groupby('Success').mean())

# creating dummy variables for categorical variables
CryoData['Female'] = (CryoData.Sex==2).astype(int)
CryoData['Plantar'] = (CryoData.Type==2).astype(int)
CryoData['Both'] = (CryoData.Type==3).astype(int)


# Data for logistic regression
CryoFeatures = np.array(CryoData.loc[:,['Age', 'Time', 'NumWarts', 'Area',
                                        'Female', 'Plantar', 'Both']])
CryoTargets = np.array(CryoData.loc[:,'Success'])
featureNames = ['Age', 'Time', 'NumWarts', 'Area', 'Female', 'Plantar', 'Both']
targetNames = ['Failure','Success']


# spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(CryoFeatures,
                                                    CryoTargets,
                                                    test_size=0.30,
                                                    random_state=0)

# Fitting the logistic regression to the training data
cryoLR = LogisticRegression()
cryoLR.fit(X_train,y_train)

# Printing out the odds ratios
print('Feature    \tOdds Ratio')
for i,iFeature in enumerate(featureNames):
    print('%-12s' % iFeature, end='')
    # Odds ratio associated with each feature (unit increase)
    print('\t%8.3f' % np.exp(cryoLR.coef_[0,i]))


# Classification on the testing data
y_pred = cryoLR.predict(X_test)

# Confusion matrix
print(confusion_matrix(y_test,y_pred))

# classification report
print(classification_report(y_test, y_pred, target_names=targetNames))
