import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# loadin the data
seedData = pd.read_csv('seeds_dataset.txt', sep='\t', header=None)
seedFeatures = np.array(seedData.iloc[:,:7])
seedTargets = np.array(seedData.iloc[:,7])
targetNames = ['Kama','Rosa','Canadian']
targetColors = ['red','blue','green']

# unscaled data PCA
seedPCA = PCA(n_components=2)
seedPCs = seedPCA.fit_transform(seedFeatures)
print(seedPCA.explained_variance_ratio_)

# scaling the data, then PCA
seedFeaturesNorm = StandardScaler().fit_transform(seedFeatures)
seedNormPCA = PCA(n_components=2)
seedNormPCs = seedNormPCA.fit_transform(seedFeaturesNorm)
print(seedNormPCA.explained_variance_ratio_)
