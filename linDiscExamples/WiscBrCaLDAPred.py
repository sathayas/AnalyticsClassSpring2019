import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# loading the data, extracting features and targets
BrCaData = pd.read_csv('WiscBrCa_clean.csv')
BrCaFeatures = np.array(BrCaData.iloc[:,1:10])
BrCaTargets = np.int_(np.array(BrCaData.iloc[:,10]) /2 -1)  # 0: benign, 1: malignant
featureNames = np.array(BrCaData.columns[1:10])
targetNames = ['Benign','Malignant']


# spliting the data into training and testing data sets


# Fitting the LDA to the training data


# Classification on the testing data


# Confusion matrix


# classification report
