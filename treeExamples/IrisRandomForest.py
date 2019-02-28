import numpy as np
import graphviz
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Loading data
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=2018)


# random forest classifier, training & testing
rf = RandomForestClassifier(criterion='entropy',
                            n_estimators = 50,
                            min_samples_leaf = 3,
                            max_depth = 4,
                            random_state=0)
rf.fit(X_train,y_train)


# evaluating the classifier performance
y_pred = rf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred,
                            target_names=target_names))

# visualizing a couple of decision trees
rfsample = rf.estimators_[5]
dot_data = export_graphviz(rfsample, feature_names=feature_names,
                           class_names=target_names, 
                           filled=True, rounded=True,  
                           special_characters=True, 
                           out_file=None)
graph = graphviz.Source(dot_data)
### Only works on Jupyter notebook. Otherwise I have to create a separate file
graph

rfsample = rf.estimators_[10]
dot_data = export_graphviz(rfsample, feature_names=feature_names,
                           class_names=target_names, 
                           filled=True, rounded=True,  
                           special_characters=True, 
                           out_file=None)
graph = graphviz.Source(dot_data)
### Only works on Jupyter notebook. Otherwise I have to create a separate file
graph





# For a comparision, 
# decision tree classifier, training & testing
dt = DecisionTreeClassifier(criterion='entropy', 
                            min_samples_leaf = 3,
                            max_depth = 4,
                            random_state=0)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred,
                            target_names=target_names))


# visualizing the decision tree
dot_data = export_graphviz(dt, feature_names=feature_names,
                           class_names=target_names, 
                           filled=True, rounded=True,  
                           special_characters=True, 
                           out_file=None)
graph = graphviz.Source(dot_data)
### Only works on Jupyter notebook. Otherwise I have to create a separate file
graph
