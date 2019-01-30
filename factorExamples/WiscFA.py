import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis

# loadin the data
wiscData = pd.read_csv('wiscsem.txt', sep='\t')
wiscFeatures = np.array(wiscData.iloc[:,2:13])
featureNames = np.array(wiscData.columns[2:13])

# applying factor analysis
FA = FactorAnalysis()  # creating a factor analysis transformation object
wiscFA = FA.fit(wiscFeatures) # fit the data

# Scree plot
eigenV = np.sum(FA.components_**2, axis=1)
plt.plot(np.arange(1,12), eigenV)
plt.xlabel('Component number')
plt.ylabel('Eigenvalue')
plt.show()

# applying factor analysis with 2 components only
reFA = FactorAnalysis(n_components=2)  # creating a factor analysis transformation object
wiscReFA = reFA.fit(wiscFeatures) # fit the data

# printing out the factor loading
print('Feature \tFactor 1\tFactor 2')
for i, iFeature in enumerate(featureNames):
    print('%-8s' % iFeature + '\t%8.3f' % reFA.components_[0,i]
          + '\t%8.3f' % reFA.components_[1,i])
