import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis

# loadin the data
pTraitData = pd.read_csv('personality0.txt', sep=' ')
numFeatures = 32

# applying factor analysis
#
# Write your own code here
#

# Scree plot
#
# Write your own code here
#

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

