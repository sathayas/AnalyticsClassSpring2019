import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# Loading the iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names


# spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# Fitting the LDA to the training data
irisLDA = LinearDiscriminantAnalysis(n_components=2)
X_train_LDA = irisLDA.fit_transform(X_train,y_train)


# Classification on the testing data
X_test_LDA = irisLDA.transform(X_test)
y_pred = irisLDA.predict(X_test)


# Plotting the training and testing data
plt.scatter(X_train_LDA[:,0],X_train_LDA[:,1],c=y_train,marker='+') # training data
plt.scatter(X_test_LDA[:,0],X_test_LDA[:,1],c=y_pred,marker='^') # testing data
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('Linear discriminant analysis')
plt.show()


# Confusion matrix
confusion_matrix(y_test,y_pred)

# classification report
print(classification_report(y_test, y_pred, target_names=target_names))
