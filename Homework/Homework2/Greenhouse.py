import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA,PCA
from sklearn.preprocessing import StandardScaler

# loading the data
ghgData = pd.read_csv('GHG_Data.txt', header=None, sep=' ')
ghgDataT = ghgData.T  # so that columns are stations
ghgDataTNorm = StandardScaler().fit_transform(ghgDataT)

# PCA first to determine the number of components
ghgPCA = PCA()
ghgPC = ghgPCA.fit_transform(ghgDataTNorm)


# cumulative explained variance ratio plot
plt.plot(np.arange(1,ghgPCA.n_components_+1),
         np.cumsum(ghgPCA.explained_variance_ratio_))
plt.xlabel('Component number')
plt.ylabel('Proportion variance explained')
plt.show()


# the number of components to achieve >90% explained variance
nIC = np.min(np.where(np.cumsum(ghgPCA.explained_variance_ratio_)>.9))+1
# there are 35 components


# ICA with the determined number of compoents
ica = FastICA(n_components=nIC)
X_ica = ica.fit_transform(ghgDataT)  


# plotting the ICs
for iIC in range(nIC):
    plt.plot(range(1,ghgDataT.shape[0]+1), X_ica[:,iIC],
             label='IC ' + str(iIC+1))
plt.legend()
plt.show()
