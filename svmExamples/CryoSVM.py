import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# loading the data
CryoData = pd.read_csv('Cryotherapy.csv')

# plotting the Age, Time, and Area
plt.figure(figsize=[9,3])
pColor = ['r','g']
plt.subplot(131)
for i in range(2):
    plt.plot(CryoData[CryoData.Success==i].Age,
             CryoData[CryoData.Success==i].Time,
             marker='o',ls='none',c=pColor[i])
plt.xlabel('Age')
plt.ylabel('Time')

plt.subplot(132)
for i in range(2):
    plt.plot(CryoData[CryoData.Success==i].Age,
             CryoData[CryoData.Success==i].Area,
             marker='o',ls='none',c=pColor[i])
plt.xlabel('Age')
plt.ylabel('Area')

plt.subplot(133)
for i in range(2):
    plt.plot(CryoData[CryoData.Success==i].Time,
             CryoData[CryoData.Success==i].Area,
             marker='o',ls='none',c=pColor[i])
plt.xlabel('Time')
plt.ylabel('Area')

plt.subplots_adjust(wspace=0.4, bottom=0.15)
plt.show()


X = CryoData.loc[:,['Age','Time','Area']]
y = CryoData.Success

# Creating the training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0)


# Exercise code here!
