import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# loading the digits data
digits = datasets.load_digits()
digitsX = digits.data    # the data, 1797 x 64 array
digitsTargets = digits.target # target information
digitsFeatureNames = digits.target_names  # digits

# Performing the linear discriminant analysis
digitsLDA = LinearDiscriminantAnalysis()
X_LDA = digitsLDA.fit_transform(digitsX,digitsTargets)

# plotting the Scree plot
plt.plot(np.arange(1,len(digitsLDA.explained_variance_ratio_)+1),
         digitsLDA.explained_variance_ratio_)
plt.xlabel('Number of components')
plt.ylabel('Variance explained')
plt.show()

# plotting the cumulative explained variance
plt.plot(np.arange(1,len(digitsLDA.explained_variance_ratio_)+1),
         np.cumsum(digitsLDA.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Variance explained')
plt.show()


# Plotting the first and second LDs
plt.scatter(X_LDA[:,0],X_LDA[:,1],c=digitsTargets,marker='+')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('Linear discriminant analysis')
plt.show()
