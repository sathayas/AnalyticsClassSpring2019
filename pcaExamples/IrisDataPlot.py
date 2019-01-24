# Plotting the iris data

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# loading the iris data set
iris = datasets.load_iris()
X = iris.data   # 2D data array
varNames = iris.feature_names  # variable names
nVar = X.shape[1]  # number of features

# plotting the data
plt.figure(figsize=[6, 6])

# generating subplots
for iRow in range(nVar-1):  # subplot row index
    for iCol in range(iRow+1, nVar):  # subplot column index
        indPlot = (nVar-1)*iRow + iCol
        plt.subplot(nVar-1, nVar-1, indPlot)
        plt.plot(X[:,iCol], X[:,iRow], 'b.')
        if iCol==iRow+1:
            plt.xlabel(varNames[iCol])
            plt.ylabel(varNames[iRow])

# adjusting the space between subplots
plt.subplots_adjust(wspace=0.35, hspace=0.35)


plt.show()
