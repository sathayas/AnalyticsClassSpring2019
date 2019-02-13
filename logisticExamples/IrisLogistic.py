import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# Loading the iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# means according types
irisDF = pd.DataFrame(np.hstack([X,np.array([y]).T]),
                      columns=iris.feature_names + ['target'])
print(irisDF.groupby('target').mean())


# spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=0)

# Fitting the logisic regression to the training data
irisLR = LogisticRegression()
irisLR.fit(X_train,y_train)

# Classification on the testing data
y_pred = irisLR.predict(X_test)
y_prob = irisLR.predict_proba(X_test)


# plotting the probabilities for different classes 
targetColors = ['magenta', 'blue', 'green']
obsVec = np.arange(1,len(y_prob)+1)
plt.figure(figsize=[8,4])
for iClass in range(3):
    plt.plot(obsVec,y_prob[:,iClass], 
             ls='-', c=targetColors[iClass])
    plt.plot(obsVec[y_pred==iClass],np.ones_like(obsVec[y_pred==iClass]),
             marker = '^', ls='none', c=targetColors[iClass], 
             label=target_names[iClass])
plt.ylim([-0.05, 1.05])
plt.xlim([1,75])
plt.xlabel('Observations')
plt.ylabel('Predicted probability')
plt.title('Predicted probabilities and classification outcome')
plt.legend(loc=0)
plt.show()


# Confusion matrix
print(confusion_matrix(y_test,y_pred))

# classification report
print(classification_report(y_test, y_pred, target_names=target_names))
