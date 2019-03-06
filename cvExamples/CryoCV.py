import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# loading the data
CryoData = pd.read_csv('Cryotherapy.csv')

# Creating the data set
X = np.array(CryoData.loc[:,['Age','Time','Area']])
y = np.array(CryoData.Success)
targetNames = ['Failure', 'Success']


# A pipeline of stadardization and kNN classifier
kNN = make_pipeline(StandardScaler(), 
                    KNeighborsClassifier(15, weights='uniform'))

# 5-fold cross validation
scores = cross_val_score(kNN, X, y, cv=5)
print(scores)

# average score
print(np.mean(scores))
