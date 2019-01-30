import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA


# loading the digits data
digits = datasets.load_digits()
digitsX = digits.data    # the data, 1797 x 64 array
digitsImages = digits.images  # image data, 1797 x 8 x 8
digitsTargets = digits.target # target information
digitsFeatureNames = digits.target_names  # digits


# PCA with all components
digitsPCA = PCA(n_components=64)
digitsPCs = digitsPCA.fit_transform(digitsX)

# Plotting explained variance ratio
plt.plot(np.arange(1,65), digitsPCA.explained_variance_ratio_)
plt.xlabel('Components')
plt.ylabel('Explained variance')
plt.show()

# Plotting cumulative explained variance ratio
plt.plot(np.arange(1,65), np.cumsum(digitsPCA.explained_variance_ratio_))
plt.xlabel('Components')
plt.ylabel('Cumulative explained variance')
plt.show()



