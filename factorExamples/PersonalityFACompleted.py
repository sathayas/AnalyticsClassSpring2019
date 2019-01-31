import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis

# loadin the data
pTraitData = pd.read_csv('personality0.txt', sep=' ')
numFeatures = 32

# applying factor analysis
FA = FactorAnalysis()  # creating a factor analysis transformation object
pTraitFA = FA.fit(pTraitData) # fit the data

# Scree plot
eigenV = np.sum(FA.components_**2, axis=1)
plt.plot(np.arange(1,numFeatures+1), eigenV)
plt.xlabel('Component number')
plt.ylabel('Eigenvalue')
plt.show()


# applying factor analysis again
numFactors = ### Fill in based on your answer to Exercise 2.
reFA = FactorAnalysis(n_components=numFactors)  # creating a factor analysis transformation object
pTraitReFA = reFA.fit(pTraitData) # fit the data

# printing out the factor loading
print('Feature  ', end='')
for iFactor in range(numFactors):
    print('\tFactor ' + str(iFactor+1), end='')
print()
for iFeature in range(numFeatures):
    print('%-8s' % pTraitData.columns[iFeature], end='')
    for iFactor in range(7):
        print('\t%8.3f' % reFA.components_[iFactor,iFeature], end='')
    print()

