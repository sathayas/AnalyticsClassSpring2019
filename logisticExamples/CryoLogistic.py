import matplotlib.pyplot as plt
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


# Exercise code here!
