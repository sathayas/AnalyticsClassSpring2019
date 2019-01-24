import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA

# loading the iris data
iris = datasets.load_iris()
X = iris.data   # 2D data array


# applying PCA
pca = PCA(n_components=2)  # creating a PCA transformation with 2 PCs
X_r = pca.fit_transform(X) # fit the data, get 2 PCs

# proportion of the variance explained
print(pca.explained_variance_ratio_)

# PCs are uncorrelated
print(np.corrcoef(X_r, rowvar=False))

# plotting PCs
plt.plot(X_r[:,0], X_r[:,1],'b.')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# plotting PCs with the target
Y = iris.target
targetNames = iris.target_names
targetColors = ['magenta', 'blue', 'green']
for iTarget in range(3):
    plt.plot(X_r[Y==iTarget,0],X_r[Y==iTarget,1], marker='+', ls='none',
         c=targetColors[iTarget], label=targetNames[iTarget])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

    
