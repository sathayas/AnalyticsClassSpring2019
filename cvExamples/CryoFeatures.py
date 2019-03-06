import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, f_classif

# loading the data
CryoData = pd.read_csv('Cryotherapy.csv')

# features, categorical and continuous
xCat = CryoData[['Sex','Type']]
xCont = CryoData[['Age','Time','NumWarts','Area']]
y = CryoData.Success

# categorical features
chiStat, chiP = chi2(xCat,y)
print(chiP)

# continuous features
fStat, fP = f_classif(xCont,y)
print(fP)
