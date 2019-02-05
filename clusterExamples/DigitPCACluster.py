import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# loading the digits data
digits = datasets.load_digits()
digitsX = digits.data    # the data, 1797 x 64 array
digitsImages = digits.images  # image data, 1797 x 8 x 8
digitsTargets = digits.target # target information
digitsFeatureNames = digits.target_names  # digits

# PCA with all possible components
digitsPCA = PCA(n_components=64)
digitsPCs = digitsPCA.fit_transform(digitsX)

# plotting the explained variance ratio (a.k.a., Scree plot)
plt.plot(np.arange(1,65),digitsPCA.explained_variance_ratio_)
plt.xlabel('Component number')
plt.ylabel('Proportion of explained variance')
plt.show()

#### Exercise Solutions Here ####
#
#
#
#

# plotting the PCs
plt.scatter(digitsPCs[:,0], digitsPCs[:,1], c=digitsTargets, marker='+')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
