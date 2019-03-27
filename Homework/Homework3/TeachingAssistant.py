import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV


# loading the data
taData = pd.read_csv('tae.csv', header=None)
taData = taData.sample(frac=1).reset_index(drop=True)  # shuffling the data
taData.columns = ['English','Instructor','Course','Semester','ClassSize','TAEval']
X = np.array(taData.iloc[:,:-1])  # features
y = np.array(taData.iloc[:,-1])   # target
feature_names = list(taData.columns[:-1])



# grid search parameters
param = {'max_depth':list(range(2,20)),
         'min_samples_leaf':list(range(2,10))}
# decision tree classifier
dt = DecisionTreeClassifier(criterion='entropy')
# grid search
grid_dt = GridSearchCV(dt, param, cv=5)
grid_dt.fit(X,y)

# the winning combination
print(grid_dt.best_params_)

# the best score
print(grid_dt.best_score_)



