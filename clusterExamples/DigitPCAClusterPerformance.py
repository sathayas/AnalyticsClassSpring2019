import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score



# loading the digits data
digits = datasets.load_digits()
digitsX = digits.data    # the data, 1797 x 64 array
digitsImages = digits.images  # image data, 1797 x 8 x 8
digitsTargets = digits.target # target information
digitsFeatureNames = digits.target_names  # digits

# PCA with all possible components
digitsPCA = PCA(n_components=64)
digitsPCs = digitsPCA.fit_transform(digitsX)

# I will go with 12 PCs
PC = digitsPCs[:,:12]

# K-means clustering with 10 clusters
km = KMeans(n_clusters=10)  
km.fit(PC)  # fitting the principal components
y_clus = km.labels_   # clustering info resulting from K-means

# ARI
print('ARI=',adjusted_rand_score(digitsTargets, y_clus),sep='')

# AMI
print('AMI=',adjusted_mutual_info_score(digitsTargets, y_clus),sep='')
