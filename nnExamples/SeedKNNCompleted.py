import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score

# loading the data 
seedData = pd.read_csv('seeds_dataset.txt', sep='\t', header=None)
seedFeatures = np.array(seedData.iloc[:,:7])
seedTargets = np.array(seedData.iloc[:,7]) - 1 # starting from zero
targetNames = ['Kama','Rosa','Canadian']
seedFeaturesNorm = StandardScaler().fit_transform(seedFeatures)

# splitting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(seedFeaturesNorm,
                                                    seedTargets, 
                                                    test_size=70, 
                                                    random_state=0)



# Standardizing the data
stdData = StandardScaler().fit(X_train)  # standardization object (based on training data)
X_train_norm = stdData.transform(X_train)  # the actual transformation
X_test_norm = stdData.transform(X_test)  # standardizing test data


# precision over k
k = np.arange(4,31,2)
precK = []
for iK in k:
    # k nearest neighbor classifier object
    kNN = KNeighborsClassifier(iK, weights='uniform')
    kNN.fit(X_train_norm,y_train)
    # predicted outcome
    y_pred = kNN.predict(X_test_norm)
    # recording the precision
    precK.append(precision_score(y_test,y_pred, average='weighted'))

                 
# plotting the precision over k
plt.plot(k,precK,'bo-')
plt.xlabel('k')
plt.ylabel('Precision')
plt.title('Precision over k')
plt.show()

