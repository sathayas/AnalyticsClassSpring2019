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


# examples of the digits data
plt.figure(figsize=(8,3))
for iImg in range(3):
    for jImg in range(10):
        plt.subplot(3,10,iImg*10+jImg+1)
        plt.imshow(digitsImages[iImg*10+jImg], cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()


# PCA with largest 2 PCs
digitsPCA = PCA(n_components=2)
digitsPCs = digitsPCA.fit_transform(digitsX)


# plotting the PCs
targetColors=['red','gold','seagreen','blue','fuchsia',
              'orange','lime','cyan','salmon','navy']
for i in range(10):
    plt.plot(digitsPCs[digitsTargets==i,0],
             digitsPCs[digitsTargets==i,1],
             marker='.', ls='none', c=targetColors[i],
             label=digitsFeatureNames[i])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# PC1 eigenimage
plt.imshow(digitsPCA.components_[0,:].reshape([8,8]),cmap=plt.cm.gray_r,
           interpolation='nearest')
plt.xticks(())
plt.yticks(())
plt.show()

# PC2 eigenimage
plt.imshow(digitsPCA.components_[1,:].reshape([8,8]),cmap=plt.cm.gray_r,
           interpolation='nearest')
plt.xticks(())
plt.yticks(())
plt.show()


