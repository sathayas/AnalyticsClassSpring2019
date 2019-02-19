import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# loading the data 
seedData = pd.read_csv('seeds_dataset.txt', sep='\t', header=None)
seedFeatures = np.array(seedData.iloc[:,:7])
seedTargets = np.array(seedData.iloc[:,7]) - 1 # starting from zero
targetNames = ['Kama','Rosa','Canadian']


# splitting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(seedFeatures,
                                                    seedTargets, 
                                                    test_size=70, 
                                                    random_state=0)

# standardizing the training and testing data sets
normTrain = StandardScaler().fit(X_train)
X_train_norm = normTrain.transform(X_train)
X_test_norm = normTrain.transform(X_test)

# Exercise codes here!
