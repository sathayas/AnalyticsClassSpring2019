import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


# loadin the data
leafData = pd.read_csv('LeafData.csv', header=None)
leafFeatures = np.array(leafData.iloc[:,1:])
leafFeaturesNorm = StandardScaler().fit_transform(leafFeatures)
leafTarget = np.array(leafData.iloc[:,0])

# PCA
leafPCA = PCA()
leafPC = leafPCA.fit_transform(leafFeaturesNorm)


# Scree plot
plt.plot(np.arange(1,leafPCA.n_components_+1),
         leafPCA.explained_variance_ratio_, 'b.-')
plt.xlabel('Component number')
plt.ylabel('Proportion variance explained')
plt.show()


#### I will use 9 components
PC = leafPC[:,:9]


# K-means clustering with 10 clusters
km = KMeans(n_clusters=10)  
km.fit(PC)  # fitting the principal components
y_clus = km.labels_   # clustering info resulting from K-means


# clustering performance
print('ARI=',adjusted_rand_score(leafTarget, y_clus),sep='')
print('AMI=',adjusted_mutual_info_score(leafTarget, y_clus),sep='')
