import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# loading the data and standarsizing
seedData = pd.read_csv('seeds_dataset.txt', sep='\t', header=None)
seedFeatures = np.array(seedData.iloc[:,:7])
seedFeaturesNorm = StandardScaler().fit_transform(seedFeatures)

# PCA
seedPCA = PCA(n_components=7)
seedPCs = seedPCA.fit_transform(seedFeaturesNorm)

# Scree plot
plt.plot(np.arange(1,8), seedPCA.explained_variance_ratio_)
plt.xlabel('Component number')
plt.ylabel('Proportion variance explained')
plt.show()

# Cumulative explained variance
plt.plot(np.arange(1,8), np.cumsum(seedPCA.explained_variance_ratio_))
plt.xlabel('Component number')
plt.ylabel('Proportion variance explained')
plt.show()
