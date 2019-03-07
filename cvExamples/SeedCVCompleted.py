import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle

# loading the data 
seedDataRaw = pd.read_csv('seeds_dataset.txt', sep='\t', header=None)
seedData = shuffle(seedDataRaw)
seedFeatures = np.array(seedData.iloc[:,:7])
seedTargets = np.array(seedData.iloc[:,7]) - 1 # starting from zero
targetNames = ['Kama','Rosa','Canadian']


# A pipeline of stadardization and kNN classifier
kNN = make_pipeline(StandardScaler(), 
                    KNeighborsClassifier(15, weights='uniform'))


# 5-fold cross validation
scores = cross_val_score(kNN, seedFeatures, seedTargets, cv=5)
print(scores)

# average score
print(np.mean(scores))

