import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# loadin the data
wiscData = pd.read_csv('wiscsem.txt', sep='\t')
wiscFeatures = np.array(wiscData.iloc[:,2:13])
featureNames = np.array(wiscData.columns[2:13])

# applying PCA
pca = PCA()  # creating a PCA transformation object
wiscPCs = pca.fit_transform(wiscFeatures) # fit the data

## We will use first two components
PC = wiscPCs[:,:2]

## We will go with 3 clusters
hc = AgglomerativeClustering(n_clusters=3)  # defining the clustering object
hc.fit(PC)  # actually fitting the data
y_clus = hc.labels_   # clustering info resulting from hieararchical


### plotting the clusters
plt.scatter(PC[:,0],PC[:,1],c=y_clus,marker='+')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters from Hierarchical')
plt.show()

