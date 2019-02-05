import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA,PCA

# loadin the data
stockData = pd.read_csv('StockMarket.csv')
stockX = np.array(stockData.iloc[:,1:])   # Data 
stockFeature = np.array(stockData.columns[1:])  # Feature names

# plotting the data
plt.figure(figsize=[8,4])
for i in range(8):
    plt.plot(stockX[:60,i], label = stockFeature[i])
plt.xlim([0,80])
plt.legend()
plt.show()

# PCA first to determine the number of components
pca = PCA()
stockPC = pca.fit_transform(stockX)

# Cumulative explained variance
plt.plot(np.arange(1,9), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Component number')
plt.ylabel('Proportion variance explained')
plt.show()

nIC = np.min(np.where(np.cumsum(pca.explained_variance_ratio_)>.9))+1

# ICA
ica = FastICA(n_components=nIC)
stockIC = ica.fit_transform(stockX)


# plotting the ICs
plt.figure(figsize=[8,4])
for iIC in range(nIC):
    plt.plot(stockIC[:60,iIC], label = 'IC ' + str(iIC+1))
plt.xlim([0,80])
plt.legend()
plt.show()

