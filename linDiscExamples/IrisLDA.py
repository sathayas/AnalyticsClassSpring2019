import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Loading the iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Performing the linear discriminant analysis
irisLDA = LinearDiscriminantAnalysis(n_components=2)
X_LDA = irisLDA.fit_transform(X,y)

# Plotting the first and second LDs
plt.scatter(X_LDA[:,0],X_LDA[:,1],c=y,marker='+')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('Linear discriminant analysis')
plt.show()

# Contribution of features
# LD1
print(irisLDA.scalings_[:,0])
# LD2
print(irisLDA.scalings_[:,1])


### Just as a comparison, PCA with the same number of components
irisPCA = PCA(n_components=2)
X_PCA = irisPCA.fit_transform(X)

plt.figure(figsize=[8,4])
# Plotting the first and second LDs
plt.subplot(121)
plt.scatter(X_LDA[:,0],X_LDA[:,1],c=y,marker='+')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('Linear discriminant analysis')

# Plotting the first and second PCs
plt.subplot(122)
plt.scatter(X_PCA[:,0],X_PCA[:,1],c=y,marker='+')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Principal component analysis')
plt.show()


# Contribution of features
# LD1
print(irisLDA.scalings_[:,0])
# LD2
print(irisLDA.scalings_[:,1])

