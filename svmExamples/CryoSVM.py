import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


# loading the data
CryoData = pd.read_csv('Cryotherapy.csv')

# Creating the data set, and normalizing it
X = np.array(CryoData.loc[:,['Age','Time','Area']])
X_norm = StandardScaler().fit_transform(X)   # standardizing it
y = np.array(CryoData.Success)


# plotting the Age, Time, and Area
plt.figure(figsize=[9,3])
pColor = ['r','g']
plt.subplot(131)
for i in range(2):
    plt.plot(X_norm[y==i,0], X_norm[y==i,1],
             marker='o',ls='none',c=pColor[i])
plt.xlabel('Age')
plt.ylabel('Time')

plt.subplot(132)
for i in range(2):
    plt.plot(X_norm[y==i,0], X_norm[y==i,2],
             marker='o',ls='none',c=pColor[i])
plt.xlabel('Age')
plt.ylabel('Area')

plt.subplot(133)
for i in range(2):
    plt.plot(X_norm[y==i,1], X_norm[y==i,2],
             marker='o',ls='none',c=pColor[i])
plt.xlabel('Time')
plt.ylabel('Area')

plt.subplots_adjust(wspace=0.4, bottom=0.15)
plt.show()


# Creating the training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y,
                                                    test_size=0.3,
                                                    random_state=0)


# Exercise code here!
