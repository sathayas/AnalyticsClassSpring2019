import numpy as np
import pandas as pd
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# loading the data, creating binary target
BrCaData = pd.read_csv('WiscBrCa_clean.csv')
BrCaData['Malignant'] = BrCaData.Class //2 -1  # Binary variable of malignancy
                                               # 0: benign
                                               # 1: malignant


# means according to malignancy
print(BrCaData.groupby('Malignant').mean())


# features and targets
BrCaFeatures = np.array(BrCaData.iloc[:,1:10])
BrCaTargets = np.array(BrCaData.Malignant)
featureNames = np.array(BrCaData.columns[1:10])
targetNames = ['Benign','Malignant']


# Exercise code here!
